"""Issue #2476 — turn-averaged SAE basis: pod-side phase-dispatch driver.

Plan: tasks/<status>/2476/plans/plan.md (v4). Phases (plan §4):
  smoke      P0 unified 1-cell smoke (composed in unit 3 of the pre-split build).
  assemble   P1 stage + assemble arm-c inputs: stream the 1,920 #779 capture
             chunks (parent's bounded per-chunk retry), extract LAYER-19
             columns only (layers-field asserted, position-of-19 indexing),
             build X19/Y19 fp16 memmaps in the parent's assembled row space
             (pass_b 5,000 rows first, then new rows in stream order);
             re-assert the parent's pinned splits by sha (split_1482.json +
             scratch_meta index arrays); print + reconcile realized row counts
             (the plan §7 kill-criterion re-measure).
  recapture  P2 arm-b recapture: teacher-forced forwards over the m-round's
             exact 30,000 rows (split_indices_matryoshka.npz @ the pinned HF
             revision, shas re-asserted vs the committed m_split.json), reusing
             the m-round token stream + hook machinery (EA._tokenize_row /
             EA._batched_capture), extended to dense answer-span means at
             layers 19 AND 20 (plain mean primary + token_inlier_mask masked
             twin); stream-reduce, per-group consolidated shards. Gates:
             G2b (hook-alignment: lmsys-dict token FVE L20-maximal vs L19/L21
             on a pilot slice, BEFORE the main loop), G2a (recaptured c20 vs
             banked dense_l20 c20, row-matched cos >= 0.999 flat bar), D2c
             (diagnostic: recaptured vbar19 vs banked v_x@19 cos distribution;
             median < 0.99 halt-investigate).
  upload1    P3 recapture store + assembled-split metadata -> HF
             issue2476_turnavg/analysis_tensors/{recapture_store,split_meta}/
             BEFORE any fit (#825 ordering); fail-loud verify.
  sae_train / maps / eval / figures — built in units 2/3 (registered stubs
             that raise NotImplementedError; never silent passes).

Pod-side contract: sentinels under /workspace/logs/issue-2476-*.json ONLY
(never task.py); [phase=...] log lines. LMSYS/WildChat text is handled
DIGEST-ONLY (never printed/logged). Resume provenance (plan §10): every
phase's resume-skip is REGIME-KEYED — a regime.json beside the outputs
carries the pinned split shas + code SHA + config hash; a skip is honored
only on full match; a code-SHA-only mismatch RECOMPUTES loudly (never skips,
per plan §10 "bare output existence never vouches after a crash-fix round");
any OTHER mismatch fails LOUD (out-root collision, #1333 / #722 r3). Resume
keys hash GENERATING PARAMETERS only — never recomputed float arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM smoke)

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue1482_early_layer as EL  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import issue1482_matryoshka_tier as M  # noqa: E402
import issue1482_sae as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2476")

TASK_ID = 2476
LAYER_C = 19  # arm-c layer: the #779 map's native layer (plan §11)
LAYERS_B = (19, 20)  # arm-b recapture layers (vbar19 diagnostic + vbar20 bridge)
HOOK_PROBE_LAYERS = (19, 21)  # G2b: lmsys-dict token FVE must be L20-maximal vs these
L_TIER = M.L_TIER  # 20 — the m-round dictionary layer
PILOT_N_G2B = 24  # G2b pilot slice size (m_pilot recipe class)
N_DENSE_SHARDS = 64  # dense_l20_g*.npz shard count (plan §2, Hub-probed at plan time)
STREAM_FLUSH_EVERY = int(os.environ.get("EPM_I2476_STREAM_FLUSH_EVERY", "50"))

# m-round store @ the pinned revision (plan §10 HF reuse rows; #1345 revision-pin rule)
M_STORE_PREFIX = "issue1482_error_analysis/analysis_tensors/matryoshka_tier/store"
M_STORE_REVISION = "8b82eb975326774003512db1e7e542491edcd056"
COMMITTED_SPLIT_1482 = PROJECT_ROOT / "eval_results" / "issue_1482" / "split_1482.json"
COMMITTED_M_SPLIT = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "matryoshka_tier" / "m_split.json"
)

# G2a bar (plan §7): flat span/single-position bf16 identity bar — calibration source:
# gotchas.md bf16 entries (#779, #1005) + the m-round's committed same-surface reference
# g2m_row_cos_min = 0.999881 (eval_results/issue_1482/matryoshka_tier/m_pilot.json).
G2A_COS_MIN = 0.999
# D2c halt-investigate floor (plan §7 D2c: DIAGNOSTIC, not a validity gate — the m-round
# token-id convention differs from #779's full-template convention at seam tokens).
D2C_MEDIAN_FLOOR = 0.99
RC_G2A = 22  # G2a identity-gate structural HALT (m-round RC class convention)
RC_HOOK = 23  # G2b hook-alignment off-by-one signature
RC_D2C = 24  # D2c halt-investigate (recaptured-vs-banked v19 median below floor)


# ── small utils ──────────────────────────────────────────────────────────────────


def _write_json(path: Path, obj: dict, *, phase: str) -> None:
    """Atomic JSON write with reproducibility metadata (git_provenance + dirty flag
    + card phase identity, per code-style.md § Reproducibility metadata)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    obj = dict(obj)
    md = C.reproducibility_metadata()
    md.update(as_metadata_dict(git_provenance(), phase=phase))
    obj.setdefault("metadata", md)
    C.write_json_atomic(path, obj)


def _sentinel(name: str, note: str, extra: dict | None = None) -> None:
    """Non-blocking phase sentinel (poller-parseable; never kills the run on OSError)."""
    payload = {"blocks_pipeline": False}
    if extra:
        payload.update(extra)
    try:
        C.write_sentinel(f"phase-{name}", note, task_id=TASK_ID, extra=payload)
    except OSError as e:
        logger.warning("[sentinel] phase-%s write failed: %s", name, e)


def _assemble_dir(args) -> Path:
    return args.out_root / "assemble"


def _recapture_dir(args) -> Path:
    return args.out_root / "recapture"


def _stage_dir(args) -> Path:
    return args.out_root / "stage"


def _sentinels_dir(args) -> Path:
    return args.out_root / "sentinels"


def _committed_split() -> dict:
    return json.loads(COMMITTED_SPLIT_1482.read_text())


def _committed_m_split() -> dict:
    return json.loads(COMMITTED_M_SPLIT.read_text())


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized per-row cosine in fp32 (no per-row python loop)."""
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    assert a32.shape == b32.shape and a32.ndim == 2, (a32.shape, b32.shape)
    num = (a32 * b32).sum(axis=1)
    den = np.linalg.norm(a32, axis=1) * np.linalg.norm(b32, axis=1)
    return num / np.maximum(den, np.float32(1e-30))


def _row_vbars(h19: torch.Tensor, h20: torch.Tensor, context_end: int) -> dict:
    """Pure per-row reduce: plain answer-span means at L19/L20 + the
    token_inlier_mask masked twin at L20 (EA._row_features keep-mask convention:
    full-sequence-median inlier mask + BOS strip; fallback to the unmasked mean
    when the mask empties the answer span, flagged)."""
    assert h19.ndim == 2 and h19.shape == h20.shape, (tuple(h19.shape), tuple(h20.shape))
    assert 0 <= context_end < h20.shape[0] - 1, (context_end, tuple(h20.shape))
    ans19 = h19[context_end + 1 :]
    ans20 = h20[context_end + 1 :]
    vbar19 = ans19.mean(0)
    vbar20 = ans20.mean(0)
    keep = S.token_inlier_mask(h20)
    keep[: min(S.BOS_OFFSET, keep.shape[0])] = False
    ans_keep = keep[context_end + 1 :]
    fallback = int(int(ans_keep.sum()) == 0)
    vbar20_in = vbar20 if fallback else ans20[ans_keep].mean(0)
    assert vbar19.shape == vbar20.shape == vbar20_in.shape == (h20.shape[1],)
    return {
        "vbar19": vbar19,
        "vbar20": vbar20,
        "vbar20_inlier": vbar20_in,
        "inlier_fallback": fallback,
    }


# ── regime-keyed resume (plan §10 Resume provenance) ─────────────────────────────


def _regime(args) -> dict:
    """Regime manifest: pinned split shas + code SHA + config hash. Keys are
    GENERATING PARAMETERS only (never recomputed float arrays — gotchas.md
    float-last-bit rule)."""
    from explore_persona_space.orchestrate.provenance import git_provenance

    committed = _committed_split()
    m_committed = _committed_m_split()
    base = {
        "smoke": bool(args.smoke),
        "tiny_model": bool(args.tiny_model),
        "max_chunks": int(args.max_chunks),
        "smoke_rows": int(args.smoke_rows),
        "layer_c": LAYER_C,
        "layers_b": list(LAYERS_B),
        "m_store_revision": M_STORE_REVISION,
        "consolidate_chunks": int(M.CONSOLIDATE_CHUNKS),
        "split_shas": {
            "pinned_val_sha256": committed["pinned_val_sha256"],
            "pinned_test_sha256": committed["pinned_test_sha256"],
            "train_full_sha256": committed["train_full_sha256"],
            "holdout_sha256": committed["holdout"]["sha256"],
            "sae_fit_sha256": committed["sae_fit"]["sha256"],
            "m_s_fit_sha256": m_committed["realized"]["s_fit_sha256"],
            "m_s_score_sha256": m_committed["realized"]["s_score_sha256"],
        },
    }
    cfg_hash = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]
    prov = git_provenance()
    code_sha = prov.commit_sha_full or prov.commit_sha or "unknown"
    return {**base, "config_hash": cfg_hash, "code_sha": code_sha}


def _enter_phase_regime(out_dir: Path, args, phase: str) -> tuple[dict, bool]:
    """Write/verify the phase regime manifest. Returns (regime, resume_ok).

    resume_ok=True  -> a FULL-match manifest exists; output-existence skips are honored.
    resume_ok=False -> fresh root, or code-SHA-only mismatch (recompute LOUDLY —
                       plan §10 "any mismatch => recompute, never skip").
    Config/split-sha mismatch -> RuntimeError (out-root regime collision, #1333).
    """
    regime = _regime(args)
    path = out_dir / "regime.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if (
            prev.get("config_hash") != regime["config_hash"]
            or prev.get("split_shas") != regime["split_shas"]
        ):
            raise RuntimeError(
                f"[{phase}] out-root {out_dir} holds a run under a DIFFERENT regime "
                f"(config_hash {prev.get('config_hash')} != {regime['config_hash']}); "
                "use a fresh --out-root (never silently mix regimes)"
            )
        if prev.get("code_sha") != regime["code_sha"]:
            logger.warning(
                "[%s] code SHA changed (%s -> %s): completed outputs under %s are "
                "RECOMPUTED, never skipped (plan §10 resume provenance)",
                phase,
                str(prev.get("code_sha"))[:12],
                regime["code_sha"][:12],
                out_dir,
            )
            _write_json(path, regime, phase=phase)
            return regime, False
        return regime, True
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, regime, phase=phase)
    return regime, False


# ── P1: stage + assemble (arm-c inputs) ──────────────────────────────────────────


def _capture_chunk_names(args) -> list[str]:
    """Sorted capture-chunk universe under the parent's own prefix constant
    (EA.CAPTURE_PREFIX — never a re-typed literal; gotchas.md reused-module rule)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    names = hub.retry_transient(
        lambda: sorted(
            f.path.rsplit("/", 1)[-1]
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO, path_in_repo=EA.CAPTURE_PREFIX, repo_type="dataset", recursive=True
            )
            if getattr(f, "size", None) is not None and f.path.endswith(".pt")
        ),
        what=f"capture chunk listing ({EA.CAPTURE_PREFIX})",
    )
    if not names:
        raise FileNotFoundError(f"no capture chunks under HF {EA.CAPTURE_PREFIX}")
    if args.max_chunks > 0:
        names = names[: args.max_chunks]
    return names


def _extract_chunk_l19(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Layer-19 columns of one capture chunk. The chunk's ``layers`` field is
    asserted == CAPTURE_LAYERS and indexed by position-of-19 (plan §12 assumption
    1 — never blind positional trust)."""
    b = F._mmap_load(path)
    layers = [int(x) for x in b["layers"]]
    assert layers == list(N1G.CAPTURE_LAYERS), (
        f"chunk {path.name} layers field {layers} != {list(N1G.CAPTURE_LAYERS)}"
    )
    assert LAYER_C in layers, (layers, LAYER_C)
    cx = N50._slice_layer(b, "cx_last", LAYER_C)
    vx = N50._slice_layer(b, "v_x", LAYER_C)
    ci = [int(x) for x in b["ci"]]
    del b
    assert cx.shape == vx.shape == (len(ci), C.EXPECTED_HIDDEN), (cx.shape, vx.shape, len(ci))
    return cx, vx, ci


def _write_cursor(path: Path, fp: str, regime: dict, cursor_chunk: int, cursor_row: int) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "fingerprint": fp,
                "config_hash": regime["config_hash"],
                "code_sha": regime["code_sha"],
                "cursor_chunk": int(cursor_chunk),
                "cursor_row": int(cursor_row),
            }
        )
    )
    os.replace(tmp, path)


def _load_scratch_meta(args) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Stage + load the parent's scratch_meta (row_ci / prov / split_indices) and
    sha-assert every pinned pool (plan §12 assumption 3 verify)."""
    ns = SimpleNamespace(scratch=_stage_dir(args))
    ns.scratch.mkdir(parents=True, exist_ok=True)
    EL._stage_scratch_meta(ns)
    pools = EL._load_split_and_assert(ns)  # sae_fit + holdout sha-asserted vs literals
    committed = _committed_split()
    z = np.load(ns.scratch / "split_indices.npz")
    for k in ("train_full", "sae_val"):
        assert k in z.files, f"split_indices.npz missing key {k}"
    train_full = np.asarray(z["train_full"], dtype=np.int64)
    sae_val = np.asarray(z["sae_val"], dtype=np.int64)
    assert EL._sha_ids(train_full) == committed["train_full_sha256"], "train_full sha drift"
    assert EL._sha_ids(sae_val) == committed["sae_val"]["sha256"], "sae_val sha drift"
    pools = {**pools, "train_full": train_full, "sae_val": sae_val}
    row_ci = np.load(ns.scratch / "row_ci.npy")
    prov_u8 = np.load(ns.scratch / "prov.npy")
    n_total = int(committed["n_total"])
    assert len(row_ci) == len(prov_u8) == n_total, (len(row_ci), len(prov_u8), n_total)
    n_pb = int(N1M.N_PASS_B)
    assert (row_ci[:n_pb] == -1).all(), "pass_b rows must lead the assembled row space (ci=-1)"
    assert (row_ci[n_pb:] >= 0).all(), "new rows must carry non-negative ci"
    return row_ci, prov_u8, pools


def _assert_pinned_valtest(committed: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive the pass_b r1_train/val/test split and sha-assert vs the pinned
    committed literals (plan P1: re-assert the parent's pinned splits by sha)."""
    r1_train, val, test = F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    got_val, got_test = F._sha_ids(val), F._sha_ids(test)
    assert got_val == committed["pinned_val_sha256"], (
        f"pinned val sha {got_val} != committed {committed['pinned_val_sha256']}"
    )
    assert got_test == committed["pinned_test_sha256"], (
        f"pinned test sha {got_test} != committed {committed['pinned_test_sha256']}"
    )
    return np.asarray(r1_train, np.int64), np.asarray(val, np.int64), np.asarray(test, np.int64)


def phase_assemble(args) -> None:
    """P1: stream the capture chunks -> X19/Y19 fp16 memmaps (parent row space)
    + rows_present.npy + split_meta.json, with realized-count reconciliation."""
    C.phase("assemble")
    out = _assemble_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    regime, resume_ok = _enter_phase_regime(out, args, "assemble")
    outputs = [out / "X19.fp16.npy", out / "Y19.fp16.npy", out / "rows_present.npy"]
    meta_path = out / "split_meta.json"
    if resume_ok and meta_path.exists() and all(p.exists() for p in outputs):
        logger.info("[assemble] resume: outputs present under matching regime; skip")
        return
    if not resume_ok:
        for p in [*outputs, meta_path, out / ".stream_cursor.json"]:
            if p.exists():
                logger.warning("[assemble] recompute: removing stale %s", p.name)
                p.unlink()
    EA._headroom(args.out_root, 2 if args.smoke else 18, "p1-assemble")
    stage = _stage_dir(args)

    committed = _committed_split()
    row_ci, prov_u8, pools = _load_scratch_meta(args)
    r1_train, val, test = _assert_pinned_valtest(committed)
    n_pb = int(N1M.N_PASS_B)
    rev = {int(c): j for j, c in enumerate(row_ci[n_pb:], start=n_pb)}

    pb = N1G._load_pass_b_bundle(N1G.PASS_B_LOCAL)
    assert LAYER_C in [int(x) for x in pb["layers"]], "pass_b bundle lacks layer 19"
    pb_x19 = N50._slice_layer(pb, "cx_last", LAYER_C)
    pb_y19 = N50._slice_layer(pb, "v_x", LAYER_C)
    del pb
    assert pb_x19.shape == pb_y19.shape == (n_pb, C.EXPECTED_HIDDEN), (pb_x19.shape, n_pb)

    names = _capture_chunk_names(args)
    fp = N1M._stream_ckpt_fingerprint(LAYER_C, EA.CAPTURE_PREFIX, names)
    if args.max_chunks > 0:
        rows_present = _assemble_stream_smoke(args, out, stage, names, rev, pb_x19, pb_y19)
    else:
        n_total = int(committed["n_total"])
        assert n_total == n_pb + int(committed["n_new_captured"]), (
            n_total,
            committed["n_new_captured"],
        )
        realized = _assemble_stream_production(
            args, out, stage, names, rev, pb_x19, pb_y19, n_total, fp, regime
        )
        # ── the plan §7 kill-criterion reconciliation: realized vs pinned counts ──
        assert realized == n_total, (
            f"[assemble] realized rows {realized} != committed n_total {n_total} — "
            "irreconcilable with the pinned split shas (plan §7 kill criterion)"
        )
        rows_present = np.arange(n_total, dtype=np.int64)
        np.save(out / "rows_present.npy", rows_present)

    n_real = int(len(rows_present))
    n_train_full = int(len(pools["train_full"]))
    print(
        f"[assemble] realized rows: pass_b={n_pb} new={n_real - n_pb} total={n_real}; "
        f"reconciliation: {n_real} vs committed n_total={committed['n_total']} "
        f"= train_full {n_train_full} + val {len(val)} + test {len(test)}",
        flush=True,
    )
    hold = pools["holdout"]
    hold_lm = int((prov_u8[hold] == 0).sum())
    hold_wc = int((prov_u8[hold] == 1).sum())
    print(
        f"[assemble] holdout per-corpus split: lmsys={hold_lm} wildchat={hold_wc} "
        f"(committed {committed['holdout']['n_lmsys']}/{committed['holdout']['n_wildchat']})",
        flush=True,
    )
    if args.max_chunks == 0:
        assert n_real == n_train_full + len(val) + len(test), (n_real, n_train_full)
        assert hold_lm == int(committed["holdout"]["n_lmsys"]), (hold_lm, committed["holdout"])
        assert hold_wc == int(committed["holdout"]["n_wildchat"]), (hold_wc, committed["holdout"])

    _write_json(
        meta_path,
        {
            "layer": LAYER_C,
            "dtype": "float16",
            "hidden": int(C.EXPECTED_HIDDEN),
            "n_pass_b": n_pb,
            "n_new": n_real - n_pb,
            "n_total_realized": n_real,
            "n_train_full": n_train_full,
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "n_r1_train": int(len(r1_train)),
            "committed_n_total": int(committed["n_total"]),
            "holdout_corpus": {"n_lmsys": hold_lm, "n_wildchat": hold_wc},
            "shas": {
                "train_full_sha256": EL._sha_ids(pools["train_full"]),
                "holdout_sha256": EL._sha_ids(pools["holdout"]),
                "sae_fit_sha256": EL._sha_ids(pools["sae_fit"]),
                "sae_val_sha256": EL._sha_ids(pools["sae_val"]),
                "pinned_val_sha256": F._sha_ids(val),  # SHA_PIN_DOMAIN: INDEX
                "pinned_test_sha256": F._sha_ids(test),  # SHA_PIN_DOMAIN: INDEX
            },
            "chunk_universe": {"n_chunks": len(names), "fingerprint": fp},
            "files": {
                "x": "X19.fp16.npy",
                "y": "Y19.fp16.npy",
                "rows_present": "rows_present.npy",
            },
            "regime": regime,
        },
        phase="assemble",
    )
    _sentinel("assemble", f"P1 done ({n_real} rows @ layer {LAYER_C})")
    logger.info("[assemble] done: %d rows", n_real)


def _assemble_stream_production(
    args, out, stage, names, rev, pb_x19, pb_y19, n_total, fp, regime
) -> int:
    """Production stream: fp16 memmaps at the KNOWN committed shape, per-chunk
    progress lines, cursor-sidecar checkpointing (resume from the cursor)."""
    xp, yp = out / "X19.fp16.npy", out / "Y19.fp16.npy"
    cur_path = out / ".stream_cursor.json"
    n_pb = int(N1M.N_PASS_B)
    start_chunk, cursor, resumed = 0, n_pb, False
    if xp.exists() and yp.exists() and cur_path.exists() and not args.fresh_stream:
        side = json.loads(cur_path.read_text())
        if (
            side.get("fingerprint") == fp
            and side.get("config_hash") == regime["config_hash"]
            and side.get("code_sha") == regime["code_sha"]
        ):
            start_chunk, cursor = int(side["cursor_chunk"]), int(side["cursor_row"])
            resumed = True
            logger.info("[assemble] RESUMED stream at chunk %d row %d", start_chunk, cursor)
        else:
            logger.warning(
                "[assemble] stream cursor MISMATCHED (fingerprint/config/code); re-streaming"
            )
    if resumed:
        x_mm = np.lib.format.open_memmap(xp, mode="r+")
        y_mm = np.lib.format.open_memmap(yp, mode="r+")
        assert x_mm.shape == y_mm.shape == (n_total, C.EXPECTED_HIDDEN), (x_mm.shape, n_total)
        assert x_mm.dtype == y_mm.dtype == np.float16
    else:
        x_mm = np.lib.format.open_memmap(
            xp, mode="w+", dtype=np.float16, shape=(n_total, C.EXPECTED_HIDDEN)
        )
        y_mm = np.lib.format.open_memmap(
            yp, mode="w+", dtype=np.float16, shape=(n_total, C.EXPECTED_HIDDEN)
        )
        x_mm[:n_pb] = pb_x19.astype(np.float16)
        y_mm[:n_pb] = pb_y19.astype(np.float16)
        start_chunk, cursor = 0, n_pb
        _write_cursor(cur_path, fp, regime, 0, cursor)
    cache = stage / "chunk_cache"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k in range(start_chunk, len(names)):
        got = Path(
            N1M._download_chunk_with_retry(C.HF_DATA_REPO, f"{EA.CAPTURE_PREFIX}/{names[k]}", cache)
        )
        cx, vx, ci = _extract_chunk_l19(got)
        got.unlink()
        rows_g = np.fromiter((rev[c] for c in ci), dtype=np.int64, count=len(ci))
        assert (rows_g == np.arange(cursor, cursor + len(ci))).all(), (
            f"[assemble] chunk {names[k]} rows not sequential at cursor {cursor} — "
            "stream order diverges from the parent's assembled row order"
        )
        x_mm[cursor : cursor + len(ci)] = cx.astype(np.float16)
        y_mm[cursor : cursor + len(ci)] = vx.astype(np.float16)
        cursor += len(ci)
        print(
            f"[assemble] chunk {k + 1}/{len(names)} rows={cursor} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        if (k + 1) % STREAM_FLUSH_EVERY == 0:
            x_mm.flush()
            y_mm.flush()
            _write_cursor(cur_path, fp, regime, k + 1, cursor)
    x_mm.flush()
    y_mm.flush()
    _write_cursor(cur_path, fp, regime, len(names), cursor)
    return cursor


def _assemble_stream_smoke(args, out, stage, names, rev, pb_x19, pb_y19) -> np.ndarray:
    """Smoke stream (--max-chunks > 0): same per-chunk extraction, rows buffered
    in RAM (bounded by max_chunks), exact-shape memmaps written once. The
    at-scale cursor/checkpoint path is production-only (plan §4 smoke
    blind-spot enumeration item 3)."""
    n_pb = int(N1M.N_PASS_B)
    cache = stage / "chunk_cache"
    cache.mkdir(parents=True, exist_ok=True)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rows_new: list[int] = []
    t0 = time.time()
    for k, name in enumerate(names):
        got = Path(
            N1M._download_chunk_with_retry(C.HF_DATA_REPO, f"{EA.CAPTURE_PREFIX}/{name}", cache)
        )
        cx, vx, ci = _extract_chunk_l19(got)
        got.unlink()
        xs.append(cx)
        ys.append(vx)
        rows_new.extend(rev[c] for c in ci)
        print(
            f"[assemble] chunk {k + 1}/{len(names)} rows={n_pb + len(rows_new)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    rows_new_arr = np.asarray(rows_new, dtype=np.int64)
    assert (np.diff(rows_new_arr) > 0).all(), "smoke stream rows not strictly increasing"
    rows_present = np.concatenate([np.arange(n_pb, dtype=np.int64), rows_new_arr])
    n = int(len(rows_present))
    x_mm = np.lib.format.open_memmap(
        out / "X19.fp16.npy", mode="w+", dtype=np.float16, shape=(n, C.EXPECTED_HIDDEN)
    )
    y_mm = np.lib.format.open_memmap(
        out / "Y19.fp16.npy", mode="w+", dtype=np.float16, shape=(n, C.EXPECTED_HIDDEN)
    )
    x_mm[:n_pb] = pb_x19.astype(np.float16)
    y_mm[:n_pb] = pb_y19.astype(np.float16)
    x_mm[n_pb:] = np.concatenate(xs).astype(np.float16)
    y_mm[n_pb:] = np.concatenate(ys).astype(np.float16)
    x_mm.flush()
    y_mm.flush()
    np.save(out / "rows_present.npy", rows_present)
    return rows_present


# ── P2: arm-b recapture ──────────────────────────────────────────────────────────


def _stage_m_split(args) -> tuple[np.ndarray, np.ndarray]:
    """Stage split_indices_matryoshka.npz @ the pinned m-store revision and
    sha-assert vs the committed m_split.json realized shas (read at run time)."""
    from explore_persona_space.orchestrate import hub

    dest = _stage_dir(args) / "split_indices_matryoshka.npz"
    if not (dest.exists() and dest.stat().st_size > 0):
        hub.stage_hub_file(
            C.HF_DATA_REPO,
            f"{M_STORE_PREFIX}/split_indices_matryoshka.npz",
            dest,
            repo_type="dataset",
            revision=M_STORE_REVISION,
        )
    m_committed = _committed_m_split()
    z = np.load(dest)
    for k in ("s_fit", "s_score"):
        assert k in z.files, f"split_indices_matryoshka.npz missing key {k}"
    s_fit = np.asarray(z["s_fit"], dtype=np.int64)
    s_score = np.asarray(z["s_score"], dtype=np.int64)
    got_fit, got_score = EL._sha_ids(s_fit), EL._sha_ids(s_score)
    assert got_fit == m_committed["realized"]["s_fit_sha256"], (
        f"m-split s_fit sha {got_fit} != committed m_split.json realized"
    )
    assert got_score == m_committed["realized"]["s_score_sha256"], (
        f"m-split s_score sha {got_score} != committed m_split.json realized"
    )
    assert len(s_fit) == int(m_committed["realized"]["n_fit"]), (len(s_fit), m_committed)
    assert len(s_score) == int(m_committed["realized"]["n_score"]), (len(s_score), m_committed)
    logger.info("[recapture] m-split sha-verified (%d fit / %d score)", len(s_fit), len(s_score))
    return s_fit, s_score


def _stage_banked_c20(args, needed_rows: np.ndarray) -> dict[int, np.ndarray]:
    """Stage the m-store dense_l20 shards @ the pinned revision; return
    row_idx -> banked c20 (fp16) for the needed rows only (~30k x 3584)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dest = _stage_dir(args) / "dense_l20"
    dest.mkdir(parents=True, exist_ok=True)
    files = hub.retry_transient(
        lambda: sorted(
            f.path
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO,
                path_in_repo=M_STORE_PREFIX,
                repo_type="dataset",
                revision=M_STORE_REVISION,
                recursive=False,
            )
            if f.path.rsplit("/", 1)[-1].startswith("dense_l20_g") and f.path.endswith(".npz")
        ),
        what=f"dense_l20 shard listing ({M_STORE_PREFIX})",
    )
    assert len(files) == N_DENSE_SHARDS, (
        f"dense_l20 shard count {len(files)} != {N_DENSE_SHARDS} (plan §2 Hub-probed value)"
    )
    needed_sorted = np.sort(np.asarray(needed_rows, dtype=np.int64))
    banked: dict[int, np.ndarray] = {}
    for p in files:
        tgt = dest / p.rsplit("/", 1)[-1]
        if not (tgt.exists() and tgt.stat().st_size > 0):
            hub.stage_hub_file(
                C.HF_DATA_REPO, p, tgt, repo_type="dataset", revision=M_STORE_REVISION
            )
        with np.load(tgt) as z:
            ridx = np.asarray(z["row_idx"], dtype=np.int64)
            mask = np.isin(ridx, needed_sorted)
            if mask.any():
                c20 = z["c20"][mask]
                for r, vec in zip(ridx[mask], c20, strict=True):
                    banked[int(r)] = vec
    logger.info("[recapture] banked c20 staged: %d/%d rows", len(banked), len(needed_sorted))
    return banked


def _pilot_rows_g2b(dns, names, needed_ci, tok, prefix_chars) -> list[tuple]:
    """First PILOT_N_G2B tokenizable rows in chunk order (deterministic pilot
    slice for the G2b hook-alignment probe)."""
    rows: list[tuple] = []
    for _name, keep in EA._iter_needed_rows(dns, names, needed_ci):
        for row_idx, ci, prompt, response in keep:
            tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
            if tk is None:
                continue
            rows.append((row_idx, ci, *tk))
            if len(rows) >= PILOT_N_G2B:
                return rows
    return rows


def _gate_g2b(args, model, tok, dns, names, needed_ci, prefix_chars) -> dict:
    """G2b hook-alignment probe: lmsys-dict token-level FVE must be L20-maximal
    vs L19/L21 on a pilot slice (m_pilot hook_fve_lmsys recipe)."""
    if args.tiny_model:
        logger.warning("[recapture] G2b SKIPPED under --tiny-model (random weights)")
        return {"verdict": "SKIPPED-tiny-model"}
    pilot = _pilot_rows_g2b(dns, names, needed_ci, tok, prefix_chars)
    assert pilot, "G2b: no tokenizable pilot rows found"
    probe_layers = tuple(sorted({L_TIER, *HOOK_PROBE_LAYERS}))
    caps = EA._batched_capture(model, tok, pilot, probe_layers, args.device)
    S.SAELensJumpReLU.ensure_downloaded(M.SAE_IDS["lmsys"], args.sae_dir)
    sae = S.SAELensJumpReLU.load(M.SAE_IDS["lmsys"], device=args.device, cache_dir=args.sae_dir)
    hook_fve: dict[str, float] = {}
    for li in probe_layers:
        h_li = torch.cat([c[li][S.BOS_OFFSET :] for c in caps])
        v, _l0, _diag = sae.fve_l0(h_li)
        hook_fve[f"L{li}"] = round(float(v), 4)
    max_at_20 = hook_fve[f"L{L_TIER}"] >= max(hook_fve[f"L{li}"] for li in HOOK_PROBE_LAYERS)
    verdict = "PASS" if max_at_20 else ("INFORMATIONAL-smoke" if args.smoke else "FAIL")
    del sae
    return {
        "hook_fve_lmsys": hook_fve,
        "max_at_20": bool(max_at_20),
        "n_pilot": len(pilot),
        "verdict": verdict,
    }


def _flush_vbar_shard(rec: dict[str, list], path: Path) -> None:
    """Atomic per-group shard write (tmp name keeps the .npz suffix — the
    np.savez suffix-append trap, #1092)."""
    arrays = {
        "row_idx": np.asarray(rec["row_idx"], np.int64),
        "set_tag": np.asarray(rec["set_tag"], np.int8),
        "n_ans": np.asarray(rec["n_ans"], np.int32),
        "inlier_fallback": np.asarray(rec["inlier_fallback"], np.int8),
        "c20_cos": np.asarray(rec["c20_cos"], np.float32),
        "vbar19": (
            np.stack(rec["vbar19"])
            if rec["vbar19"]
            else np.empty((0, C.EXPECTED_HIDDEN), np.float16)
        ),
        "vbar20": (
            np.stack(rec["vbar20"])
            if rec["vbar20"]
            else np.empty((0, C.EXPECTED_HIDDEN), np.float16)
        ),
        "vbar20_inlier": (
            np.stack(rec["vbar20_inlier"])
            if rec["vbar20_inlier"]
            else np.empty((0, C.EXPECTED_HIDDEN), np.float16)
        ),
    }
    tmp = path.parent / f".tmp_{path.name}"
    np.savez(tmp, **arrays)
    tmp.replace(path)


def phase_recapture(args) -> None:
    """P2: teacher-forced recapture of the m-round's 30k rows at layers 19+20;
    per-group shards + consolidation + gates (G2b pre-loop, G2a/D2c post)."""
    C.phase("recapture")
    out = _recapture_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    regime, resume_ok = _enter_phase_regime(out, args, "recapture")
    store_path = out / "vbar_store.npz"
    gates_path = out / "gates_p2.json"
    if resume_ok and store_path.exists() and gates_path.exists():
        logger.info("[recapture] resume: store + gates present under matching regime; skip")
        return
    if not resume_ok:
        for p in [store_path, gates_path, *out.glob("vbar_g*.npz")]:
            if p.exists():
                logger.warning("[recapture] recompute: removing stale %s", p.name)
                p.unlink()
    EA._headroom(args.out_root, 1 if args.smoke else 6, "p2-recapture")
    stage = _stage_dir(args)
    ns = SimpleNamespace(scratch=stage)
    ns.scratch.mkdir(parents=True, exist_ok=True)
    EL._stage_scratch_meta(ns)
    row_ci = np.load(stage / "row_ci.npy")

    s_fit, s_score = _stage_m_split(args)
    dns = SimpleNamespace(max_chunks=args.max_chunks, scratch=stage)
    if args.max_chunks > 0:  # smoke SCALE knob — restrict to the enumerated raw chunks
        universe = EL._chunk_ci_universe(dns, EA._raw_chunk_names(dns))
        s_fit = s_fit[np.isin(row_ci[s_fit], list(universe))]
        s_score = s_score[np.isin(row_ci[s_score], list(universe))]
        logger.info(
            "[recapture] chunk-restricted rows (max_chunks=%d): fit %d, score %d",
            args.max_chunks,
            len(s_fit),
            len(s_score),
        )
    set_tag = {int(r): 1 for r in s_fit}
    set_tag.update({int(r): 0 for r in s_score})
    rows_all = np.asarray(sorted(set_tag), dtype=np.int64)
    if args.smoke_rows > 0:
        rows_all = rows_all[: args.smoke_rows]
        set_tag = {int(r): set_tag[int(r)] for r in rows_all}
        logger.info("[recapture] --smoke-rows cap: %d rows", len(rows_all))
    assert len(rows_all) > 0, "recapture row set is empty"
    needed_ci = {int(row_ci[r]): int(r) for r in rows_all}
    assert -1 not in needed_ci, "recapture rows must be NEW rows (text-resolvable)"

    banked_c20: dict[int, np.ndarray] = {}
    if args.tiny_model:
        logger.warning("[recapture] G2a bank compare SKIPPED under --tiny-model")
    else:
        banked_c20 = _stage_banked_c20(args, rows_all)
        if args.max_chunks == 0 and args.smoke_rows == 0:
            missing_bank = set(int(r) for r in rows_all) - set(banked_c20)
            assert not missing_bank, (
                f"banked dense_l20 missing {len(missing_bank)} of {len(rows_all)} rows "
                f"(e.g. {sorted(missing_bank)[:5]})"
            )

    model, tok = EA._load_model_tok(args)
    prefix_chars = EA._prefix_char_len(tok)
    names = EA._raw_chunk_names(dns)
    gates: dict = {"g2b": _gate_g2b(args, model, tok, dns, names, needed_ci, prefix_chars)}
    if gates["g2b"]["verdict"] == "FAIL":
        _write_json(gates_path, gates, phase="recapture")
        _sentinel("recapture", "G2b hook-alignment FAIL (gates_p2.json written)", {"rc": RC_HOOK})
        logger.error("[recapture] G2b hook-alignment NOT L20-maximal: %s", gates["g2b"])
        sys.exit(RC_HOOK)

    groups = [
        names[i : i + M.CONSOLIDATE_CHUNKS] for i in range(0, len(names), M.CONSOLIDATE_CHUNKS)
    ]
    n_done, tok_count = 0, 0
    t_loop = time.time()
    for gi, group in enumerate(groups):
        shard = out / f"vbar_g{gi:04d}.npz"
        if shard.exists():
            continue  # per-group resume-skip (whole group landed atomically)
        rec: dict[str, list] = {
            k: []
            for k in (
                "row_idx",
                "set_tag",
                "n_ans",
                "inlier_fallback",
                "c20_cos",
                "vbar19",
                "vbar20",
                "vbar20_inlier",
            )
        }
        for _name, keep in EA._iter_needed_rows(dns, group, needed_ci):
            rows = []
            for row_idx, ci, prompt, response in keep:
                tk = EA._tokenize_row(tok, prompt, response, prefix_chars)
                if tk is None:
                    continue
                rows.append((row_idx, ci, *tk))
            rows.sort(key=lambda r: len(r[2]))
            for s0 in range(0, len(rows), args.gen_batch):
                batch = rows[s0 : s0 + args.gen_batch]
                caps = EA._batched_capture(model, tok, batch, LAYERS_B, args.device)
                for (row_idx, _ci, full_ids, _pe, context_end, n_ans, _seam), cap in zip(
                    batch, caps, strict=True
                ):
                    h19, h20 = cap[LAYERS_B[0]], cap[LAYERS_B[1]]
                    assert h20.shape[0] == len(full_ids) == context_end + 1 + n_ans, (
                        tuple(h20.shape),
                        len(full_ids),
                        context_end,
                        n_ans,
                    )
                    vb = _row_vbars(h19, h20, context_end)
                    cos = float("nan")
                    bk = banked_c20.get(int(row_idx))
                    if bk is not None:
                        cos = float(
                            torch.nn.functional.cosine_similarity(
                                h20[context_end].float(),
                                torch.from_numpy(np.asarray(bk, np.float32)),
                                dim=0,
                            )
                        )
                    rec["row_idx"].append(int(row_idx))
                    rec["set_tag"].append(set_tag[int(row_idx)])
                    rec["n_ans"].append(int(n_ans))
                    rec["inlier_fallback"].append(vb["inlier_fallback"])
                    rec["c20_cos"].append(cos)
                    rec["vbar19"].append(vb["vbar19"].numpy().astype(np.float16))
                    rec["vbar20"].append(vb["vbar20"].numpy().astype(np.float16))
                    rec["vbar20_inlier"].append(vb["vbar20_inlier"].numpy().astype(np.float16))
                    tok_count += len(full_ids)
                    n_done += 1
        _flush_vbar_shard(rec, shard)
        print(
            f"[recapture] unit {gi + 1}/{len(groups)} rows_total={n_done} tok={tok_count} "
            f"elapsed={time.time() - t_loop:.0f}s",
            flush=True,
        )
    del model
    _consolidate_and_gate(args, out, rows_all, set_tag, gates, gates_path, store_path)


def _consolidate_and_gate(args, out, rows_all, set_tag, gates, gates_path, store_path) -> None:
    """Consolidate per-group shards -> vbar_store.npz; evaluate G2a + D2c;
    write gates_p2.json FIRST, then halt on a production gate failure."""
    shards = sorted(out.glob("vbar_g*.npz"))
    assert shards, "no recapture shards to consolidate"
    parts: dict[str, list[np.ndarray]] = {}
    for p in shards:
        with np.load(p) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    arr = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(arr["row_idx"], kind="stable")
    arr = {k: v[order] for k, v in arr.items()}
    assert (np.diff(arr["row_idx"]) > 0).all(), "duplicate row_idx in consolidated recapture store"
    captured = set(int(r) for r in arr["row_idx"])
    expected = set(int(r) for r in rows_all)
    missing = expected - captured
    extra = captured - expected
    assert not extra, f"recapture store carries {len(extra)} unexpected rows"
    production = args.max_chunks == 0 and args.smoke_rows == 0 and not args.smoke
    if production:
        assert not missing, (
            f"recapture missing {len(missing)} of {len(expected)} rows "
            f"(e.g. {sorted(missing)[:5]}) — every m-split row must capture (plan §7 G2)"
        )
    elif missing:
        logger.warning(
            "[recapture] smoke: %d/%d rows missing (informational)", len(missing), len(expected)
        )
    tmp = store_path.parent / f".tmp_{store_path.name}"
    np.savez(tmp, **arr)
    tmp.replace(store_path)

    # ── G2a: recaptured c20 vs banked dense_l20 c20 (row-matched, flat bar) ──
    cos = arr["c20_cos"]
    finite = np.isfinite(cos)
    if args.tiny_model or int(finite.sum()) == 0:
        gates["g2a"] = {"verdict": "SKIPPED-tiny-model", "n": 0}
    else:
        cvals = cos[finite]
        g2a_pass = bool(cvals.min() >= G2A_COS_MIN)
        gates["g2a"] = {
            "n": int(finite.sum()),
            "row_cos_min": float(cvals.min()),
            "row_cos_median": float(np.median(cvals)),
            "bar": G2A_COS_MIN,
            "verdict": (
                "PASS" if g2a_pass else ("INFORMATIONAL-smoke" if not production else "FAIL")
            ),
        }

    # ── D2c: recaptured vbar19 vs banked v_x@19 (diagnostic distribution) ──
    a_dir = _assemble_dir(args)
    assert (a_dir / "split_meta.json").exists(), "D2c needs the P1 assemble outputs — run assemble"
    rows_present = np.load(a_dir / "rows_present.npy")
    y19 = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    pos = np.searchsorted(rows_present, arr["row_idx"])
    ok = (pos < len(rows_present)) & (
        rows_present[np.minimum(pos, len(rows_present) - 1)] == arr["row_idx"]
    )
    if int(ok.sum()) == 0:
        gates["d2c"] = {"verdict": "SKIPPED-no-overlap", "n": 0}
        logger.warning("[recapture] D2c: 0 rows overlap the assembled slice (smoke)")
    else:
        d_cos = _cos_rows(arr["vbar19"][ok], np.asarray(y19[pos[ok]]))
        med = float(np.median(d_cos))
        gates["d2c"] = {
            "n": int(ok.sum()),
            "median": med,
            "p05": float(np.quantile(d_cos, 0.05)),
            "p25": float(np.quantile(d_cos, 0.25)),
            "p75": float(np.quantile(d_cos, 0.75)),
            "min": float(d_cos.min()),
            "floor": D2C_MEDIAN_FLOOR,
            "verdict": (
                "PASS"
                if med >= D2C_MEDIAN_FLOOR
                else ("INFORMATIONAL-smoke" if not production else "HALT-INVESTIGATE")
            ),
        }

    _write_json(gates_path, gates, phase="recapture")
    n_rows = int(len(arr["row_idx"]))
    _sentinel(
        "recapture",
        f"P2 done ({n_rows} rows; g2a={gates['g2a']['verdict']} "
        f"g2b={gates['g2b']['verdict']} d2c={gates['d2c']['verdict']})",
    )
    logger.info(
        "[recapture] gates: %s", json.dumps({k: v.get("verdict") for k, v in gates.items()})
    )
    if production and gates["g2a"].get("verdict") == "FAIL":
        logger.error("[recapture] G2a identity gate FAIL: %s", gates["g2a"])
        sys.exit(RC_G2A)
    if production and gates["d2c"].get("verdict") == "HALT-INVESTIGATE":
        logger.error("[recapture] D2c median below floor: %s", gates["d2c"])
        sys.exit(RC_D2C)


# ── P3: upload stores BEFORE fits (#825 ordering) ────────────────────────────────


def phase_upload1(args) -> None:
    """P3: recapture store + assembled-split metadata -> HF, fail-loud verify;
    regime-keyed done-file (sentinels/upload1.done.json)."""
    C.phase("upload1")
    sent_dir = _sentinels_dir(args)
    sent_dir.mkdir(parents=True, exist_ok=True)
    regime, resume_ok = _enter_phase_regime(sent_dir, args, "upload1")
    done_path = sent_dir / "upload1.done.json"
    if resume_ok and done_path.exists():
        logger.info("[upload1] resume: done-file present under matching regime; skip")
        return
    if args.skip_upload:
        logger.warning("[upload1] --skip-upload: store upload SKIPPED (local-only run)")
        _sentinel("upload1", "P3 SKIPPED (--skip-upload)")
        return
    recap = _recapture_dir(args)
    a_dir = _assemble_dir(args)
    assert (recap / "vbar_store.npz").exists() and (recap / "gates_p2.json").exists(), (
        "upload1 needs the P2 outputs — run recapture first"
    )
    assert (a_dir / "split_meta.json").exists(), "upload1 needs the P1 outputs — run assemble"

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    # split-metadata bundle (index arrays + shas + regime files, plan §4 P3)
    meta_dir = _stage_dir(args) / "split_meta_upload"
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a_dir / "split_meta.json", meta_dir / "split_meta.json")
    shutil.copy2(a_dir / "regime.json", meta_dir / "assemble_regime.json")
    shutil.copy2(recap / "regime.json", meta_dir / "recapture_regime.json")
    shutil.copy2(a_dir / "rows_present.npy", meta_dir / "rows_present.npy")
    m_split_local = _stage_dir(args) / "split_indices_matryoshka.npz"
    assert m_split_local.exists(), "staged split_indices_matryoshka.npz missing — run recapture"
    shutil.copy2(m_split_local, meta_dir / "split_indices_matryoshka.npz")
    _write_json(
        meta_dir / "m_split_asserted.json",
        {
            "source": f"{M_STORE_PREFIX}/split_indices_matryoshka.npz",
            "revision": M_STORE_REVISION,
            "committed_realized": _committed_m_split()["realized"],
            "re_asserted": True,
        },
        phase="upload1",
    )

    prefixes = {}
    for local, leaf in ((recap, "recapture_store"), (meta_dir, "split_meta")):
        prefix = f"{args.hf_prefix}/{leaf}"
        res = upload_dir_sharded(
            local,
            C.HF_DATA_REPO,
            prefix,
            repo_type="dataset",
            shard_glob="*",
            verify=True,
            delete_local=False,
            resume_skip=resume_ok,
        )
        prefixes[leaf] = {
            "prefix": prefix,
            "n_uploaded": len(res.uploaded),
            "rerouted": bool(res.rerouted),
        }
        if not res.rerouted:  # fail-loud exact-set verify (plan P3)
            expected = [
                f"{prefix}/{p.name}"
                for p in sorted(local.iterdir())
                if p.is_file() and not p.name.startswith(".")
            ]
            missing = hub.verify_repo_paths_uploaded(
                HfApi(), C.HF_DATA_REPO, expected, path_in_repo=prefix
            )
            assert not missing, f"[upload1] verify FAILED — missing on Hub: {missing}"
        logger.info("[upload1] %s -> %s (rerouted=%s)", local.name, prefix, res.rerouted)

    _write_json(done_path, {"regime": regime, "prefixes": prefixes}, phase="upload1")
    _sentinel("upload1", f"P3 done ({json.dumps({k: v['prefix'] for k, v in prefixes.items()})})")
    logger.info("[upload1] done")


# ── later-unit stubs (units 2/3) ─────────────────────────────────────────────────


def phase_smoke(args) -> None:
    raise NotImplementedError(
        "P0 unified smoke is composed in unit 3 of the pre-split build; unit-1 legs are "
        "runnable now via --phase assemble/recapture with --smoke --max-chunks/--smoke-rows"
    )


def phase_sae_train(args) -> None:
    raise NotImplementedError("P4 sae_train (MatryoshkaBatchTopKSAE trainer) is built in unit 2")


def phase_maps(args) -> None:
    raise NotImplementedError("P5 maps + baselines is built in unit 2")


def phase_eval(args) -> None:
    raise NotImplementedError("P6 encode + DVs + stats is built in unit 2")


def phase_figures(args) -> None:
    raise NotImplementedError("P7 figures + eval JSONs + upload is built in unit 3")


PHASE_ORDER = (
    "smoke",
    "assemble",
    "recapture",
    "upload1",
    "sae_train",
    "maps",
    "eval",
    "figures",
)
PHASES = {
    "smoke": phase_smoke,
    "assemble": phase_assemble,
    "recapture": phase_recapture,
    "upload1": phase_upload1,
    "sae_train": phase_sae_train,
    "maps": phase_maps,
    "eval": phase_eval,
    "figures": phase_figures,
}


# ── CLI ──────────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Issue #2476 turn-averaged SAE phase driver (see module docstring)"
    )
    ap.add_argument("--phase", default="all", choices=["all", *PHASE_ORDER])
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps_out/issue2476"))
    ap.add_argument(
        "--hf-prefix",
        default="issue2476_turnavg/analysis_tensors",
        help="HF data-repo destination prefix (issue-owned; never a parent's prefix)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny-N run of the SAME pipeline")
    ap.add_argument(
        "--tiny-model", action="store_true", help="24-layer from-config Qwen2 (CPU smoke carve-out)"
    )
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all 1,920 chunks (production)")
    ap.add_argument("--smoke-rows", type=int, default=0, help="0 = all m-split rows (production)")
    ap.add_argument("--gen-batch", type=int, default=16, help="P2 teacher-forced forward batch")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--sae-dir", type=Path, default=None, help="SAELens weight cache dir")
    ap.add_argument("--fresh-stream", action="store_true", help="P1: ignore the stream cursor")
    ap.add_argument("--skip-upload", action="store_true", help="P3: local-only run (loud)")
    ap.add_argument("--gpu-id", type=int, default=-1, help="informational; CVD pins the device")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute completeness + call-arity bind + deferred-import resolution",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Deferred-import resolution (smoke-architecture Axis 1): execute every
        # function-body import of this driver so a missing symbol fails HERE.
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

        from explore_persona_space.analysis.extraction import (
            extract_layer_activations,  # noqa: F401
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.preflight import (
            assert_out_root_headroom,  # noqa: F401
        )
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )
        from explore_persona_space.orchestrate.upload_sharded import (
            upload_dir_sharded,  # noqa: F401
        )

        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.sae_dir is None:
        args.sae_dir = args.out_root / "sae_cache"
    args.out_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[main] phase=%s out_root=%s device=%s gpu_id=%s smoke=%s",
        args.phase,
        args.out_root,
        args.device,
        args.gpu_id,
        args.smoke,
    )
    seq = PHASE_ORDER if args.phase == "all" else (args.phase,)
    for name in seq:
        PHASES[name](args)
    # explicit exit: heavy C-extension teardown must not rewrite the rc (gotchas.md)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
