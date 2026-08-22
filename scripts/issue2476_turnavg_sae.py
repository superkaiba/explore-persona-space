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
  sae_train  P4 train the arm-c MatryoshkaBatchTopKSAE on the turn-averaged
             answer summaries (Y19 rows minus holdout/val/test; 10k SAE-val
             carve seed 2476): BatchTopK k=100 (2412.06410) + matryoshka
             nested prefix losses (2503.17547), Adam(0.9,0.999) LR 2e-4,
             batch 256, 3 epochs, fp16 memmap block-shuffled streaming;
             per-epoch checkpoints + regime-keyed resume; gate G4 (SAE-val
             var-FVE >= 0.5, rc=RC_G4); weights+cfg+train_log -> HF sae_c/.
  maps       P5 arm-c refits via EA.phase_p1_fit VERBATIM (refit_full /
             refit_holdout / refit_lmsys_transfer, ridge seed0; gate G1
             reconciliation vs the committed #1482 values read at run time,
             rc=RC_G1), arm-c identity+bias (streamed bias, helper-parity
             asserted), arm-b shared-Gram ridge c20->vbar20 (+ inlier twin +
             identity+bias), dense-input companions (c19->f_true 120k/20k;
             c20->f_true20 24k/6k) — all predictions persisted for P6.
  eval       P6 gate G3 FIRST (chanind lmsys fve_l0 on the 30k turn
             averages, floor 0.35 = the parent gate_bm halt band; below ->
             arm-b DEMOTION to exploratory-with-caveat, NEVER a run abort),
             then encode f_true/f_pred/f_ib and the vectorized batteries:
             per-feature held-out R2, per-tier medians, within-activity-
             quintile tier permutation (m-round kernel), K=20 row-shuffle
             null, 10k-draw bootstrap CIs, per-tier kNN retrieval
             (map/identity+bias/train-mean), dense-space anchor + encoder
             recon diagnostics, arm-b bridge vs the committed token-level
             per-feature npz, corpus-transfer fold read.
  figures — built in unit 3 (registered stub raising NotImplementedError).

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
import math
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
import issue779_percontext_recon as PR  # noqa: E402
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


# ── P4: MatryoshkaBatchTopKSAE + trainer (arm-c dictionary) ──────────────────────

SAE_SEED = 2476  # SAE init + SAE-val carve (plan §10 Seeds)
SAE_VAL_N = 10_000
SAE_LR = 2e-4  # plan §11: arXiv 2606.28548 recipe
SAE_ADAM_BETAS = (0.9, 0.999)
SAE_BATCH = 256
SAE_EPOCHS = 3
SAE_K = 100
SAE_DICT = int(S.SAELENS_DICT_SIZE)  # 65,536 (arm-b comparability, plan §11)
SAE_THRESH_EMA = 0.999  # BatchTopK inference-threshold EMA (2412.06410 convention)
SAE_BLOCK_ROWS = 65_536  # block-shuffled epoch loader granularity (fp16 memmap streaming)
G4_FVE_FLOOR = 0.5  # plan §7 G4: structural-breakage floor (expected ~0.85)
BOOT_SEED_2476 = 24_761  # plan §10 Seeds: bootstrap
SHUFFLE_SEEDS_2476 = tuple(range(2_476_100, 2_476_120))  # K=20 row-shuffle null (advisory)
RC_G4 = 25  # G4 SAE-val FVE-floor HALT (unit-1 convention: 22/23/24 taken)
RC_G1 = 26  # G1 reconciliation HALT (plan §7: halt fits)
EA_FIT_IDS = (
    "refit_full__ridge__seed0",
    "refit_holdout__ridge__seed0",
    "refit_lmsys_transfer__ridge__seed0",  # corpus-transfer fold (plan §6, one extra Gram)
)


class MatryoshkaBatchTopKSAE(torch.nn.Module):
    """Matryoshka BatchTopK SAE over turn-averaged answer summaries (plan §4 P4).

    Training forward: ReLU pre-activations gated by BatchTopK — keep the B*k
    largest activations across the WHOLE batch (arXiv 2412.06410), selected ONCE
    on the full 65,536 width — then the matryoshka NESTED reconstruction losses
    over the feature-id prefixes (2,048 / 16,384 / 65,536; arXiv 2503.17547),
    MEAN over the three prefix losses (recorded in cfg.json). Inference: scalar
    threshold gating (EMA of each training batch's minimum kept activation — the
    BatchTopK inference convention the in-repo andyrdt loader consumes). fp32
    throughout. No decoder renorm / aux loss: BatchTopK carries no shrinkage
    incentive, and G4 (held-out var-FVE floor) is the training-sanity arbiter.
    Duck-types SAELensJumpReLU's encode/decode contract (act_dim / dict_size /
    device) so the shared encode/recon helpers below run on either dictionary."""

    def __init__(
        self,
        act_dim: int = int(C.EXPECTED_HIDDEN),
        dict_size: int = SAE_DICT,
        k: int = SAE_K,
        tier_bounds: tuple[int, ...] = S.MATRYOSHKA_TIER_BOUNDS,
        seed: int = SAE_SEED,
    ):
        super().__init__()
        assert int(tier_bounds[-1]) == dict_size, (tier_bounds, dict_size)
        gen = torch.Generator().manual_seed(seed)
        w_dec = torch.randn(dict_size, act_dim, generator=gen, dtype=torch.float32)
        w_dec = w_dec / w_dec.norm(dim=1, keepdim=True)
        self.w_dec = torch.nn.Parameter(w_dec)
        self.w_enc = torch.nn.Parameter(w_dec.t().contiguous().clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(dict_size))
        self.b_dec = torch.nn.Parameter(torch.zeros(act_dim))
        self.register_buffer("threshold", torch.zeros(()))
        self.act_dim, self.dict_size, self.k = int(act_dim), int(dict_size), int(k)
        self.tier_bounds = tuple(int(b) for b in tier_bounds)
        self.seed = int(seed)

    @property
    def device(self):  # SAELensJumpReLU duck-type (shared helpers read .device)
        return self.w_enc.device

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.w_enc + self.b_enc

    def train_step_losses(self, x: torch.Tensor) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """One training forward: BatchTopK codes + nested prefix losses.
        Returns (loss, float diagnostics, detached (dict_size,) fired mask)."""
        assert x.ndim == 2 and x.shape[1] == self.act_dim, tuple(x.shape)
        act = torch.relu(self._pre(x))
        k_tot = min(self.k * x.shape[0], act.numel())
        with torch.no_grad():
            vals, idx = torch.topk(act.detach().flatten(), k_tot)
            mask = torch.zeros(act.numel(), dtype=torch.bool, device=act.device)
            mask[idx] = vals > 0  # never keep exact zeros (all-zero rows at init)
            mask = mask.view_as(act)
            kept = vals[vals > 0]
            min_kept = float(kept.min()) if kept.numel() else 0.0
        f = act * mask  # hard 0/1 mask; gradients flow through kept activations only
        losses = []
        for m in self.tier_bounds:
            xhat = f[:, :m] @ self.w_dec[:m] + self.b_dec
            losses.append(((x - xhat) ** 2).sum(-1).mean())
        loss = torch.stack(losses).mean()
        if min_kept > 0.0:  # EMA inference threshold (min kept activation per batch)
            with torch.no_grad():
                if float(self.threshold) == 0.0:
                    self.threshold.fill_(min_kept)
                else:
                    self.threshold.mul_(SAE_THRESH_EMA).add_((1 - SAE_THRESH_EMA) * min_kept)
        diags = {
            "loss": float(loss.detach()),
            "loss_prefix": [float(v.detach()) for v in losses],
            "l0_train": float(mask.sum()) / x.shape[0],
            "min_kept": min_kept,
        }
        return loss, diags, mask.any(0).detach()

    @torch.no_grad()
    def encode(self, h: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """(T, act_dim) -> (T, dict_size) threshold-gated inference codes (fp32)."""
        assert h.ndim == 2 and h.shape[1] == self.act_dim, tuple(h.shape)
        outs = []
        for s in range(0, h.shape[0], chunk):
            x = h[s : s + chunk].to(device=self.device, dtype=torch.float32)
            act = torch.relu(self._pre(x))
            outs.append(act * (act > self.threshold))
        return torch.cat(outs) if len(outs) != 1 else outs[0]

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        assert f.ndim == 2 and f.shape[1] == self.dict_size, tuple(f.shape)
        return f.to(device=self.device, dtype=torch.float32) @ self.w_dec + self.b_dec

    def cfg_dict(self) -> dict:
        return {
            "architecture": "matryoshka_batchtopk",
            "act_dim": self.act_dim,
            "dict_size": self.dict_size,
            "k": self.k,
            "tier_bounds": list(self.tier_bounds),
            "seed": self.seed,
            "lr": SAE_LR,
            "adam_betas": list(SAE_ADAM_BETAS),
            "batch": SAE_BATCH,
            "epochs": SAE_EPOCHS,
            "prefix_loss_reduction": "mean",
            "threshold_ema": SAE_THRESH_EMA,
            "threshold": float(self.threshold),
        }

    def save_dir(self, out: Path) -> None:
        """weights (safetensors) + cfg.json — the plan §6.5 sae_c deliverables."""
        from safetensors.torch import save_file

        sd = {
            "w_enc": self.w_enc.detach().cpu().contiguous(),
            "w_dec": self.w_dec.detach().cpu().contiguous(),
            "b_enc": self.b_enc.detach().cpu().contiguous(),
            "b_dec": self.b_dec.detach().cpu().contiguous(),
            "threshold": self.threshold.detach().cpu().reshape(1),
        }
        save_file(sd, str(out / "sae_weights.safetensors"))
        _write_json(out / "cfg.json", self.cfg_dict(), phase="sae_train")

    @classmethod
    def load_local(cls, d: Path, device: str = "cpu") -> "MatryoshkaBatchTopKSAE":
        from safetensors.torch import load_file

        cfg = json.loads((d / "cfg.json").read_text())
        sd = load_file(str(d / "sae_weights.safetensors"))
        expected = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
        assert set(sd) == expected, f"sae_weights key drift: {sorted(sd)}"
        obj = cls(
            act_dim=int(cfg["act_dim"]),
            dict_size=int(cfg["dict_size"]),
            k=int(cfg["k"]),
            tier_bounds=tuple(cfg["tier_bounds"]),
            seed=int(cfg["seed"]),
        )
        with torch.no_grad():
            obj.w_enc.copy_(sd["w_enc"])
            obj.w_dec.copy_(sd["w_dec"])
            obj.b_enc.copy_(sd["b_enc"])
            obj.b_dec.copy_(sd["b_dec"])
            obj.threshold.copy_(sd["threshold"].reshape(()))
        return obj.to(device).eval()


def _sae_out_dir(args) -> Path:
    return args.out_root / "sae_c"


def _maps_dir(args) -> Path:
    return args.out_root / "maps"


def _eval_dir(args) -> Path:
    return args.out_root / "eval"


def _production(args) -> bool:
    """One production predicate for the P4-P6 gates (the P2 convention)."""
    return args.max_chunks == 0 and args.smoke_rows == 0 and not args.smoke


@torch.no_grad()
def _recon_fve(sae, mm, positions: np.ndarray, chunk: int = 2048) -> tuple[float, float]:
    """Plain var-FVE + mean L0 of the sae's reconstruction over the given rows
    (fp64 accumulators; per-dim unbiased variance, summed — the G4 mechanism.
    NO token-pool outlier drop: these are row summaries, not token streams;
    the chanind G3 read keeps its own fve_l0 semantics)."""
    pos = np.asarray(positions, np.int64)
    n = int(len(pos))
    assert n >= 2, f"var-FVE needs >= 2 rows, got {n}"
    x_sum = torch.zeros(sae.act_dim, dtype=torch.float64, device=sae.device)
    x_sq = torch.zeros_like(x_sum)
    r_sum = torch.zeros_like(x_sum)
    r_sq = torch.zeros_like(x_sum)
    l0 = 0.0
    for s in range(0, n, chunk):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
        f = sae.encode(x, chunk=chunk)
        r = x - sae.decode(f)
        x_sum += x.sum(0, dtype=torch.float64)
        x_sq += (x * x).sum(0, dtype=torch.float64)
        r_sum += r.sum(0, dtype=torch.float64)
        r_sq += (r * r).sum(0, dtype=torch.float64)
        l0 += float((f > 0).sum())

    def _var_sum(ssum: torch.Tensor, ssq: torch.Tensor) -> float:
        return float(((ssq - ssum * ssum / n) / (n - 1)).sum())

    ss_tot = _var_sum(x_sum, x_sq)
    fve = float("nan") if ss_tot < 1e-12 else 1.0 - _var_sum(r_sum, r_sq) / ss_tot
    return fve, l0 / n


def _dead_by_tier(fired: np.ndarray) -> dict:
    """Dead-feature fraction per matryoshka tier (plan §4 P4 logging duty).
    An empty tier (sub-production dict widths in smokes) reports None, not NaN."""
    tiers = S.tier_of(np.arange(len(fired)))
    out = {}
    for t in (0, 1, 2):
        m = tiers == t
        out[str(t)] = round(float((~fired[m]).mean()), 4) if int(m.sum()) else None
    return out


def _sae_row_positions(args) -> tuple[np.ndarray, np.ndarray, dict]:
    """SAE-c train/val row POSITIONS in the assembled memmap: all present rows
    minus 20k holdout minus pinned val/test (plan §4 P4; re-measured here), with
    the 10,000-row SAE-val carve at seed 2476."""
    a_dir = _assemble_dir(args)
    rows_present = np.load(a_dir / "rows_present.npy")
    committed = _committed_split()
    _row_ci, _prov, pools = _load_scratch_meta(args)
    _r1, val, test = _assert_pinned_valtest(committed)
    excl = np.union1d(np.union1d(pools["holdout"], val), test)
    pool_ids = np.setdiff1d(rows_present, excl, assume_unique=False)
    pos = np.searchsorted(rows_present, pool_ids)
    assert (rows_present[pos] == pool_ids).all()
    rng = np.random.default_rng(SAE_SEED)
    perm = rng.permutation(len(pos))
    val_n = SAE_VAL_N if len(pos) > 3 * SAE_VAL_N else max(2, len(pos) // 5)
    val_pos = np.sort(pos[perm[:val_n]])
    tr_pos = np.sort(pos[perm[val_n:]])
    expected_pool = int(committed["n_total"]) - len(pools["holdout"]) - len(val) - len(test)
    doc = {
        "n_pool": int(len(pos)),
        "n_train": int(len(tr_pos)),
        "n_val": int(val_n),
        "n_rows_present": int(len(rows_present)),
        "expected_pool_production": expected_pool,
        "carve_seed": SAE_SEED,
    }
    if _production(args):
        assert len(pos) == expected_pool, doc  # the plan §4 re-measure reconciliation
    return tr_pos, val_pos, doc


def _block_batches(mm, positions: np.ndarray, batch: int, rng):
    """Two-level shuffled batches off an fp16 memmap: shuffle BLOCK order, load one
    sorted-position block (sequential-ish IO), shuffle within, yield (b, H) fp16 —
    approximates a full epoch shuffle without materializing the matrix (plan §4
    'fp16 memmap streaming; never materialize the full matrix in RAM')."""
    n_blocks = math.ceil(len(positions) / SAE_BLOCK_ROWS)
    for bi in rng.permutation(n_blocks):
        blk = positions[bi * SAE_BLOCK_ROWS : (bi + 1) * SAE_BLOCK_ROWS]
        arr = np.asarray(mm[blk])
        w = rng.permutation(len(arr))
        for s in range(0, len(arr), batch):
            yield arr[w[s : s + batch]]


def phase_sae_train(args) -> None:
    """P4: train the arm-c MatryoshkaBatchTopKSAE on the turn-averaged answer
    summaries (Y19 rows); per-epoch checkpoints + regime-keyed resume; gate G4;
    weights+cfg+train_log uploaded to HF sae_c/ (optimizer states never uploaded
    — the plan §10 discarded-artifacts row)."""
    C.phase("sae_train")
    out = _sae_out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    regime, resume_ok = _enter_phase_regime(out, args, "sae_train")
    w_path = out / "sae_weights.safetensors"
    log_path = out / "train_log.json"
    gates_path = out / "gates_p4.json"
    if resume_ok and w_path.exists() and log_path.exists() and gates_path.exists():
        logger.info("[sae_train] resume: weights+log+gates present under matching regime; skip")
        return
    if not resume_ok:
        for p in (w_path, log_path, gates_path, out / "cfg.json", out / "ckpt_last.pt"):
            if p.exists():
                logger.warning("[sae_train] recompute: removing stale %s", p.name)
                p.unlink()
    EA._headroom(args.out_root, 1 if args.smoke else 4, "p4-sae-train")
    a_dir = _assemble_dir(args)
    assert (a_dir / "split_meta.json").exists(), "sae_train needs the P1 outputs — run assemble"
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    tr_pos, val_pos, pool_doc = _sae_row_positions(args)
    production = _production(args)
    print(f"[sae_train] pools re-measured: {json.dumps(pool_doc)}", flush=True)

    dev = args.device
    model = MatryoshkaBatchTopKSAE().to(dev)
    # b_dec init: seeded train-subsample mean (streamed fp64; standard SAE practice)
    rng0 = np.random.default_rng(SAE_SEED + 1)
    sub = np.sort(rng0.choice(tr_pos, size=min(65_536, len(tr_pos)), replace=False))
    mu = np.zeros(model.act_dim, dtype=np.float64)
    for s in range(0, len(sub), 8192):
        mu += np.asarray(y_mm[sub[s : s + 8192]], np.float64).sum(0)
    with torch.no_grad():
        model.b_dec.copy_(torch.as_tensor(mu / len(sub), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=SAE_LR, betas=SAE_ADAM_BETAS)
    ckpt_path = out / "ckpt_last.pt"
    start_epoch, step = 0, 0
    epoch_rows: list[dict] = []
    if resume_ok and ckpt_path.exists():
        # self-produced local checkpoint (regime-matched dir) — weights_only=False
        # is the sanctioned posture for sha-pinned self-produced bundles (gotchas.md)
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = int(ck["epoch_done"]), int(ck["step"])
        epoch_rows = list(ck["log_rows"])
        logger.info("[sae_train] RESUMED at epoch %d (step %d)", start_epoch, step)
    steps_cap = int(args.sae_steps)
    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, SAE_EPOCHS):
        rng_e = np.random.default_rng(SAE_SEED * 1000 + epoch)
        fired = torch.zeros(model.dict_size, dtype=torch.bool, device=dev)
        run_loss, run_n = 0.0, 0
        diags: dict = {"l0_train": float("nan")}
        for xb in _block_batches(y_mm, tr_pos, SAE_BATCH, rng_e):
            x = torch.as_tensor(np.asarray(xb, np.float32), device=dev)
            loss, diags, fired_b = model.train_step_losses(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            fired |= fired_b
            run_loss += diags["loss"]
            run_n += 1
            step += 1
            if step % 200 == 0:
                print(
                    f"[sae_train] epoch {epoch + 1}/{SAE_EPOCHS} step {step} "
                    f"loss={run_loss / max(1, run_n):.1f} thr={float(model.threshold):.4f} "
                    f"l0={diags['l0_train']:.0f} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if steps_cap and step >= steps_cap:
                stop = True
                break
        fve_val, l0_val = _recon_fve(model, y_mm, val_pos)
        row = {
            "epoch": epoch + 1,
            "steps": step,
            "mean_loss": round(run_loss / max(1, run_n), 3),
            "val_var_fve": round(fve_val, 6),
            "val_l0": round(l0_val, 2),
            "dead_frac_by_tier": _dead_by_tier(fired.cpu().numpy()),
            "threshold": float(model.threshold),
            "elapsed_s": round(time.time() - t0, 1),
        }
        epoch_rows.append(row)
        print(f"[sae_train] unit {epoch + 1}/{SAE_EPOCHS} epoch-done {json.dumps(row)}", flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch_done": epoch + 1,
                "step": step,
                "log_rows": epoch_rows,
            },
            ckpt_path,
        )
        if stop:
            break
    assert epoch_rows, "sae_train produced no epoch rows"
    fve_val = float(epoch_rows[-1]["val_var_fve"])
    g4_pass = fve_val >= G4_FVE_FLOOR
    gates = {
        "g4": {
            "val_var_fve": fve_val,
            "val_l0": epoch_rows[-1]["val_l0"],
            "floor": G4_FVE_FLOOR,
            "n_val": int(len(val_pos)),
            "verdict": "PASS" if g4_pass else ("FAIL" if production else "INFORMATIONAL-smoke"),
        }
    }
    # persist weights + log + gates BEFORE any halt (halt-investigate needs the artifact)
    model.save_dir(out)
    _write_json(
        log_path,
        {
            "pools": pool_doc,
            "epochs": epoch_rows,
            "steps": step,
            "steps_cap": steps_cap,
            "cfg": model.cfg_dict(),
        },
        phase="sae_train",
    )
    _write_json(gates_path, gates, phase="sae_train")
    if production and not g4_pass:
        _sentinel("sae_train", "G4 SAE-val FVE below floor (gates_p4.json written)", {"rc": RC_G4})
        logger.error("[sae_train] G4 FAIL (halt-investigate before any encode): %s", gates["g4"])
        sys.exit(RC_G4)
    if args.skip_upload:
        logger.warning("[sae_train] --skip-upload: sae_c upload SKIPPED (local-only run)")
    else:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub
        from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

        up = _stage_dir(args) / "sae_c_upload"
        if up.exists():
            shutil.rmtree(up)
        up.mkdir(parents=True, exist_ok=True)
        for name in ("sae_weights.safetensors", "cfg.json", "train_log.json"):
            shutil.copy2(out / name, up / name)
        prefix = f"{args.hf_prefix}/sae_c"
        res = upload_dir_sharded(
            up,
            C.HF_DATA_REPO,
            prefix,
            repo_type="dataset",
            shard_glob="*",
            verify=True,
            delete_local=False,
            resume_skip=False,
        )
        if not res.rerouted:  # fail-loud exact-set verify (the P3 pattern)
            expected = [f"{prefix}/{p.name}" for p in sorted(up.iterdir()) if p.is_file()]
            missing = hub.verify_repo_paths_uploaded(
                HfApi(), C.HF_DATA_REPO, expected, path_in_repo=prefix
            )
            assert not missing, f"[sae_train] verify FAILED — missing on Hub: {missing}"
        logger.info("[sae_train] uploaded sae_c -> %s (rerouted=%s)", prefix, res.rerouted)
    _sentinel(
        "sae_train",
        f"P4 done (fve={fve_val:.4f} l0={epoch_rows[-1]['val_l0']} g4={gates['g4']['verdict']})",
    )
    logger.info("[sae_train] done: %s", gates["g4"])


# ── P5: maps + baselines ─────────────────────────────────────────────────────────


def _ea_scratch(args, production: bool) -> Path:
    """EA-shaped scratch: X.npy/Y.npy symlinks onto the P1 memmaps + the parent
    split_indices.npz (position-REMAPPED under smoke so EA.phase_p1_fit runs
    VERBATIM on the assembled slice; production symlinks the staged original —
    row id == memmap position there by the P1 reconciliation)."""
    d = _maps_dir(args) / "ea_scratch"
    d.mkdir(parents=True, exist_ok=True)
    a_dir = _assemble_dir(args)
    for src, name in ((a_dir / "X19.fp16.npy", "X.npy"), (a_dir / "Y19.fp16.npy", "Y.npy")):
        dest = d / name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(src.resolve())
    ns = SimpleNamespace(scratch=_stage_dir(args))
    ns.scratch.mkdir(parents=True, exist_ok=True)
    EL._stage_scratch_meta(ns)
    src_split = _stage_dir(args) / "split_indices.npz"
    z = np.load(src_split)
    needed = ("train_full", "holdout", "val", "test", "train_lmsys", "sae_fit", "sae_val")
    missing = [k for k in needed if k not in z.files]
    assert not missing, f"parent split_indices.npz missing keys {missing}"
    dest_split = d / "split_indices.npz"
    if dest_split.is_symlink() or dest_split.exists():
        dest_split.unlink()
    if production:
        dest_split.symlink_to(src_split.resolve())
        return d
    # smoke: remap every pool to POSITIONS within the assembled slice; a pool with
    # < 2 surviving rows gets a seeded stand-in carve (flagged; EA's own
    # setdiff1d(train_full, holdout) keeps stand-in holdout rows out of training)
    rows_present = np.load(a_dir / "rows_present.npy")
    remapped: dict[str, np.ndarray] = {}
    standins: dict[str, int] = {}
    for k in needed:
        ids = np.asarray(z[k], np.int64)
        pos = np.searchsorted(rows_present, ids)
        ok = (pos < len(rows_present)) & (
            rows_present[np.minimum(pos, len(rows_present) - 1)] == ids
        )
        remapped[k] = np.sort(pos[ok]).astype(np.int64)
    used = np.zeros(len(rows_present), dtype=bool)
    for k in ("holdout", "val", "test"):
        used[remapped[k]] = True
    rng = np.random.default_rng(SAE_SEED + 7)
    for k in needed:
        if len(remapped[k]) >= 2:
            continue
        free = np.where(~used)[0]
        assert len(free) >= 2, f"smoke split remap: no free positions for stand-in pool {k!r}"
        take = np.sort(rng.choice(free, size=min(50, max(2, len(free) // 5)), replace=False))
        remapped[k] = take.astype(np.int64)
        if k in ("holdout", "val", "test"):
            used[take] = True
        standins[k] = int(len(take))
        logger.warning("[maps] SMOKE stand-in pool %r: %d carved positions", k, len(take))
    np.savez(dest_split, **remapped)
    _write_json(
        d / "smoke_remap.json",
        {"standins": standins, "n_positions": {k: int(len(v)) for k, v in remapped.items()}},
        phase="maps",
    )
    return d


def _run_ea_refits(args, maps_dir: Path, scratch: Path, resume_ok: bool) -> None:
    """The three arm-c ridge refits through EA.phase_p1_fit VERBATIM (plan §4 P5)."""
    pdir = maps_dir / "percontext"
    for fit_id in EA_FIT_IDS:
        jpath, npath = pdir / f"{fit_id}.json", pdir / f"{fit_id}.npz"
        if resume_ok and jpath.exists() and npath.exists():
            logger.info("[maps] %s present; resume-skip", fit_id)
            continue
        ea = SimpleNamespace(
            scratch=scratch,
            out_eval=maps_dir,
            fit_id=fit_id,
            seed=0,
            device=args.device,
            smoke=bool(args.smoke),
            fit_n=int(args.fit_n),
            krr_nystrom_centers=16384,  # parent's explicit value (unused by ridge)
        )
        t0 = time.time()
        EA.phase_p1_fit(ea)
        print(f"[maps] unit {fit_id} elapsed={time.time() - t0:.0f}s", flush=True)


def _gate_g1(args, maps_dir: Path, production: bool) -> dict:
    """G1 reconciliation (plan §7): recomputed whole-map R2 vs the committed
    values READ AT RUN TIME from eval_results/issue_1482/percontext/ (never
    retyped). Production FAIL -> recon_gate.json written FIRST, then rc=RC_G1."""
    ref_dir = PROJECT_ROOT / "eval_results" / "issue_1482" / "percontext"
    try:
        ref_full = json.loads((ref_dir / "refit_full__ridge__seed0.json").read_text())
        ref_hold = json.loads((ref_dir / "refit_holdout__ridge__seed0.json").read_text())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e} — committed #1482 reference JSONs absent; on a partial-clone pod run "
            "`git sparse-checkout add eval_results/issue_1482`"
        ) from e
    ours_full = json.loads((maps_dir / "percontext" / "refit_full__ridge__seed0.json").read_text())
    ours_hold = json.loads(
        (maps_dir / "percontext" / "refit_holdout__ridge__seed0.json").read_text()
    )
    committed_test = float(ref_full["sets"]["test"]["whole_map_r2"])
    committed_hold = float(ref_hold["sets"]["holdout"]["whole_map_r2"])
    got_test = float(ours_full["sets"]["test"]["whole_map_r2"])
    got_hold = float(ours_hold["sets"]["holdout"]["whole_map_r2"])
    tol_full = float(EA.GATE_A_TOL["ridge"])  # 0.002 — the parent's own tolerance
    tol_hold = 0.003  # plan §7 G1
    d_full, d_hold = abs(got_test - committed_test), abs(got_hold - committed_hold)
    ok = d_full <= tol_full and d_hold <= tol_hold
    gate = {
        "committed": {"full_test": committed_test, "holdout": committed_hold},
        "recomputed": {"full_test": got_test, "holdout": got_hold},
        "abs_delta": {"full_test": d_full, "holdout": d_hold},
        "tol": {"full_test": tol_full, "holdout": tol_hold},
        "verdict": "PASS" if ok else ("FAIL" if production else "INFORMATIONAL-smoke"),
    }
    _write_json(maps_dir / "recon_gate.json", {"g1": gate}, phase="maps")
    if production and not ok:
        _sentinel("maps", "G1 reconciliation FAIL (recon_gate.json written)", {"rc": RC_G1})
        logger.error("[maps] G1 FAIL — assembly/split bug; halting fits: %s", gate)
        sys.exit(RC_G1)
    return gate


def _stream_bias(X, Y, rows: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """train-mean(Y − X) in fp64, streamed — the identity_bias_predict bias
    without materializing the full fp64 train arrays (parity-asserted below)."""
    s = np.zeros(X.shape[1], np.float64)
    for i in range(0, len(rows), chunk):
        r = rows[i : i + chunk]
        s += np.asarray(Y[r], np.float64).sum(0) - np.asarray(X[r], np.float64).sum(0)
    return s / max(1, len(rows))


def _ib_arm_c(args, maps_dir: Path, scratch: Path) -> None:
    """Arm-c identity+bias baseline (plan §4 P5): v̂_ib = c19_holdout + b, with b
    = train-mean(v19 − c19) over the SAME rows as refit_holdout. The streamed
    fp64 bias is parity-asserted against the canonical helper on a bounded
    subsample (the helper materializes full fp64 arrays — the zeros probe
    extracts its bias exactly: identity_bias_predict(x, y, 0) == b)."""
    out = maps_dir / "ib_c.npz"
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    X = np.load(scratch / "X.npy", mmap_mode="r")
    Y = np.load(scratch / "Y.npy", mmap_mode="r")
    idx = np.load(scratch / "split_indices.npz")
    tr = np.sort(np.setdiff1d(idx["train_full"], idx["holdout"], assume_unique=False))
    hold = np.asarray(idx["holdout"], np.int64)
    b = _stream_bias(X, Y, tr)
    sub = tr[: min(20_000, len(tr))]
    b_helper = identity_bias_predict(
        np.asarray(X[sub]), np.asarray(Y[sub]), np.zeros((1, X.shape[1]), np.float64)
    )[0]
    b_sub = _stream_bias(X, Y, sub)
    parity = float(np.abs(b_helper - b_sub).max())
    assert np.allclose(b_helper, b_sub, atol=1e-8), f"ib bias parity failed: {parity}"
    pred = np.asarray(X[hold], np.float64) + b
    tmp = out.parent / f".tmp_{out.name}"
    np.savez(
        tmp,
        rows=hold,
        pred16=pred.astype(np.float16),
        bias=b.astype(np.float64),
        parity_max_abs=np.float64(parity),
        n_train=np.int64(len(tr)),
    )
    tmp.replace(out)
    logger.info("[maps] ib_c done (n_train=%d, parity=%.2e)", len(tr), parity)


def _eigh_fallback(fn, device: str):
    """cuda eigh non-convergence -> exact CPU-LAPACK re-run (gotchas.md cuSOLVER
    rule: numerical-backend swap, never a Gram jitter)."""
    try:
        return fn(device)
    except torch.linalg.LinAlgError as e:
        logger.warning("[maps] cuda eigh non-convergence (%s); CPU fallback", e)
        return fn("cpu")


def _gram_ridge_single(Z, Y, tr, va, te, lambdas, device: str, block: int = 0):
    """ONE parent _ridge_factorize + pooled-R2 lambda selection on va + te
    prediction, memmap-friendly for a single wide target block (the plan §4
    'all alive features solved off ONE Gram, feature-chunked' fit; parent
    internals UNCHANGED — the byte-cap mirrors EA._shared_gram_ridge_multi)."""
    block = block or N1M.RIDGE_BLOCK
    block = min(block, max(2048, int((4 * (1 << 30)) // (max(1, Y.shape[1]) * 8))))

    def _run(d):
        dev = torch.device(d)
        fac = N1M._ridge_factorize(Z, Y, tr, dev, block)
        yva = np.asarray(Y[va], np.float64)
        best = (float(lambdas[0]), -np.inf)
        for lam in lambdas:
            pv = N1M._ridge_predict_one(Z, va, fac, lam, dev, block)
            r2 = PR._pooled_r2(pv, yva)
            if np.isfinite(r2) and r2 > best[1]:
                best = (float(lam), float(r2))
        pt = N1M._ridge_predict_one(Z, te, fac, best[0], dev, block)
        edge = bool(best[0] in (float(lambdas[0]), float(lambdas[-1])))
        return pt, {"selected_lambda": best[0], "val_r2": best[1], "lambda_grid_edge": edge}

    return _eigh_fallback(_run, device)


@torch.no_grad()
def _encode_counts(sae, mm, positions: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Per-feature active-row counts of the sae's inference codes over the given
    rows (streaming; counts on TRUE-summary encodes — the alive-mask provenance,
    plan §6: never predicted codes)."""
    counts = torch.zeros(sae.dict_size, dtype=torch.int64, device=sae.device)
    pos = np.sort(np.asarray(positions, np.int64))
    t0 = time.time()
    n_chunks = math.ceil(len(pos) / chunk)
    for i, s in enumerate(range(0, len(pos), chunk)):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
        counts += (sae.encode(x, chunk=chunk) > 0).sum(0)
        if (i + 1) % 10 == 0 or i + 1 == n_chunks:
            print(
                f"[maps] counts chunk {i + 1}/{n_chunks} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    return counts.cpu().numpy()


@torch.no_grad()
def _encode_restricted(sae, mm, positions: np.ndarray, cols: np.ndarray, out_mm=None, chunk=4096):
    """Encode the given rows -> restrict to the alive cols -> fp16 (into out_mm
    when given, else RAM). Row ORDER follows `positions` verbatim."""
    pos = np.asarray(positions, np.int64)
    cols_t = torch.as_tensor(np.asarray(cols, np.int64), device=sae.device)
    dst = out_mm if out_mm is not None else np.empty((len(pos), len(cols)), np.float16)
    for s in range(0, len(pos), chunk):
        x = torch.as_tensor(np.asarray(mm[pos[s : s + chunk]], np.float32), device=sae.device)
        f = sae.encode(x, chunk=chunk)[:, cols_t]
        dst[s : s + f.shape[0]] = f.cpu().numpy().astype(np.float16)
    return dst


def _alive_floor(n_fit: int) -> int:
    """The 1% activity criterion (plan §4/§6; EA._p3_prep convention)."""
    return max(1, math.ceil(0.01 * n_fit))


def _armb_all(args, maps_dir: Path, production: bool) -> dict:
    """Arm-b sub-unit: shared-Gram ridge c20->vbar20 (+ inlier twin off the SAME
    factorization) + identity+bias + chanind alive mask/f_true encodes + the
    c20->f_true20 dense-input companion. One staging of the banked c20."""
    out_maps = maps_dir / "armb_maps.npz"
    out_alive = maps_dir / "alive_b.npz"
    out_ftrue = maps_dir / "ftrue_b.npz"
    out_dense = maps_dir / "densein_b.npz"
    if all(p.exists() for p in (out_maps, out_alive, out_ftrue, out_dense)):
        z = np.load(out_maps)
        return {"resumed": True, "selected_lambda": float(z["selected_lambda"])}
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    store = np.load(_recapture_dir(args) / "vbar_store.npz")
    row_idx = np.asarray(store["row_idx"], np.int64)
    set_tag = np.asarray(store["set_tag"], np.int8)
    vbar20 = np.asarray(store["vbar20"], np.float16)
    vbar20_in = np.asarray(store["vbar20_inlier"], np.float16)
    banked = _stage_banked_c20(args, row_idx)
    have = np.asarray([int(r) in banked for r in row_idx], bool)
    if production:
        assert have.all(), f"banked c20 missing for {int((~have).sum())} store rows"
    row_idx, set_tag = row_idx[have], set_tag[have]
    vbar20, vbar20_in = vbar20[have], vbar20_in[have]
    Z = np.stack([banked[int(r)] for r in row_idx]).astype(np.float16)
    fit_pos = np.where(set_tag == 1)[0]
    te = np.where(set_tag == 0)[0]
    assert len(fit_pos) >= 2 and len(te) >= 2, (len(fit_pos), len(te))
    carve = min(M.PROD_VAL_CARVE, max(1, len(fit_pos) // 6))
    perm = np.random.default_rng(M.CARVE_SEED).permutation(len(fit_pos))
    va, tr = fit_pos[perm[:carve]], fit_pos[perm[carve:]]
    EL._assert_estimator_validity(len(tr), Z.shape[1], args.smoke)
    res = _eigh_fallback(
        lambda d: EA._shared_gram_ridge_multi(
            Z,
            {"vbar20": vbar20, "vbar20_inlier": vbar20_in},
            tr,
            va,
            te,
            N1M.LAMBDAS_N1M,
            torch.device(d),
            N1M.RIDGE_BLOCK,
        ),
        args.device,
    )
    pt, meta = res["vbar20"]
    pt_in, meta_in = res["vbar20_inlier"]
    ib = identity_bias_predict(
        np.asarray(Z[tr], np.float64), np.asarray(vbar20[tr], np.float64), np.asarray(Z[te])
    )
    tmp = out_maps.parent / f".tmp_{out_maps.name}"
    np.savez(
        tmp,
        row_idx_score=row_idx[te],
        row_idx_fit=row_idx[fit_pos],
        pred16=pt.astype(np.float16),
        pred16_inlier=pt_in.astype(np.float16),
        ib_pred16=ib.astype(np.float16),
        selected_lambda=np.float64(meta["selected_lambda"]),
        val_r2=np.float64(meta["val_r2"]),
        selected_lambda_inlier=np.float64(meta_in["selected_lambda"]),
        n_train=np.int64(len(tr)),
        carve=np.int64(carve),
    )
    tmp.replace(out_maps)

    # chanind lmsys: alive mask on TRUE fit-side encodes + f_true over all rows
    sae = S.SAELensJumpReLU.load(M.SAE_IDS["lmsys"], device=args.device, cache_dir=args.sae_dir)
    counts = _encode_counts(sae, vbar20, fit_pos)
    floor = _alive_floor(len(fit_pos))
    alive = np.where(counts >= floor)[0].astype(np.int64)
    assert len(alive) >= 1, "no alive features (arm b)"
    f_true = _encode_restricted(sae, vbar20, np.arange(len(row_idx)), alive)
    f_true_in = _encode_restricted(sae, vbar20_in, te, alive)  # inlier twin, te rows only
    train_mean = np.asarray(f_true[fit_pos], np.float64).mean(0)
    tmp = out_alive.parent / f".tmp_{out_alive.name}"
    np.savez(
        tmp,
        alive_ids=alive,
        counts=counts.astype(np.int64),
        floor=np.int64(floor),
        n_fit_rows=np.int64(len(fit_pos)),
        train_mean=train_mean.astype(np.float32),
        tier=S.tier_of(alive),
    )
    tmp.replace(out_alive)
    tmp = out_ftrue.parent / f".tmp_{out_ftrue.name}"
    np.savez(tmp, row_idx=row_idx, f_true=f_true, f_true_inlier_te=f_true_in)
    tmp.replace(out_ftrue)

    # dense-input companion (plan §4 P5): c20 -> f_true20, same carve, ONE Gram
    pt_d, meta_d = _gram_ridge_single(Z, f_true, tr, va, te, N1M.LAMBDAS_N1M, args.device)
    tmp = out_dense.parent / f".tmp_{out_dense.name}"
    np.savez(
        tmp,
        pred16=pt_d.astype(np.float16),
        feat_ids=alive,
        rows=row_idx[te],
        selected_lambda=np.float64(meta_d["selected_lambda"]),
        val_r2=np.float64(meta_d["val_r2"]),
    )
    tmp.replace(out_dense)
    del sae
    print(f"[maps] unit armb done (alive={len(alive)}, lam={meta['selected_lambda']})", flush=True)
    return {
        "n_fit": int(len(fit_pos)),
        "n_score": int(len(te)),
        "selected_lambda": float(meta["selected_lambda"]),
        "val_r2": float(meta["val_r2"]),
        "n_alive": int(len(alive)),
        "alive_floor": int(floor),
        "densein_selected_lambda": float(meta_d["selected_lambda"]),
    }


def _dense_companion_c(args, maps_dir: Path, scratch: Path, production: bool) -> dict:
    """Arm-c SAE-c alive mask + f_true encodes (fit-side 120k + 20k holdout, ONE
    memmap) + the c19->f_true dense-input companion off ONE Gram (plan §4 P5)."""
    alive_path = maps_dir / "alive_c.npz"
    ftrue_path = maps_dir / "ftrue_c_all.fp16.npy"
    dense_path = maps_dir / "densein_c.npz"
    if all(p.exists() for p in (alive_path, ftrue_path, dense_path)):
        z = np.load(alive_path)
        return {"resumed": True, "n_alive": int(len(z["alive_ids"]))}
    sae = MatryoshkaBatchTopKSAE.load_local(_sae_out_dir(args), device=args.device)
    a_dir = _assemble_dir(args)
    X = np.load(a_dir / "X19.fp16.npy", mmap_mode="r")
    Ymm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")
    idx = np.load(scratch / "split_indices.npz")
    sae_fit = np.sort(np.asarray(idx["sae_fit"], np.int64))
    hold = np.asarray(idx["holdout"], np.int64)  # EA holdout ORDER (pred16 alignment)
    n_fit = int(len(sae_fit))
    counts = _encode_counts(sae, Ymm, sae_fit)
    floor = _alive_floor(n_fit)
    alive = np.where(counts >= floor)[0].astype(np.int64)
    assert len(alive) >= 1, "no alive features (arm c)"
    rows_c = np.concatenate([sae_fit, hold])
    yc = np.lib.format.open_memmap(
        ftrue_path, mode="w+", dtype=np.float16, shape=(len(rows_c), len(alive))
    )
    _encode_restricted(sae, Ymm, rows_c, alive, out_mm=yc)
    yc.flush()
    tm = np.zeros(len(alive), np.float64)
    for s in range(0, n_fit, 8192):
        tm += np.asarray(yc[s : s + 8192], np.float64).sum(0)
    train_mean = tm / max(1, n_fit)
    tmp = alive_path.parent / f".tmp_{alive_path.name}"
    np.savez(
        tmp,
        alive_ids=alive,
        counts=counts.astype(np.int64),
        floor=np.int64(floor),
        n_fit_rows=np.int64(n_fit),
        train_mean=train_mean.astype(np.float32),
        tier=S.tier_of(alive),
    )
    tmp.replace(alive_path)
    Xc = np.asarray(X[rows_c], np.float16)
    carve = min(M.PROD_VAL_CARVE, max(1, n_fit // 6))
    perm = np.random.default_rng(M.CARVE_SEED).permutation(n_fit)
    va, tr = perm[:carve], perm[carve:]
    te = np.arange(n_fit, len(rows_c))
    EL._assert_estimator_validity(len(tr), Xc.shape[1], args.smoke)
    pt, meta = _gram_ridge_single(Xc, yc, tr, va, te, N1M.LAMBDAS_N1M, args.device)
    tmp = dense_path.parent / f".tmp_{dense_path.name}"
    np.savez(
        tmp,
        pred16=pt.astype(np.float16),
        feat_ids=alive,
        rows=hold,
        selected_lambda=np.float64(meta["selected_lambda"]),
        val_r2=np.float64(meta["val_r2"]),
    )
    tmp.replace(dense_path)
    del sae
    print(f"[maps] unit densein_c done (alive={len(alive)})", flush=True)
    return {
        "n_alive": int(len(alive)),
        "alive_floor": int(floor),
        "n_fit_rows": n_fit,
        "selected_lambda": float(meta["selected_lambda"]),
        "val_r2": float(meta["val_r2"]),
    }


def phase_maps(args) -> None:
    """P5: arm-c EA refits (gate G1) + identity+bias + arm-b shared-Gram map +
    dense-input companions + corpus-transfer fold — predictions persisted for P6."""
    C.phase("maps")
    out = _maps_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    regime, resume_ok = _enter_phase_regime(out, args, "maps")
    meta_path = out / "preds_meta.json"
    if resume_ok and meta_path.exists():
        logger.info("[maps] resume: preds_meta present under matching regime; skip")
        return
    if not resume_ok:
        stale = [
            meta_path,
            out / "recon_gate.json",
            out / "ib_c.npz",
            out / "armb_maps.npz",
            out / "alive_b.npz",
            out / "ftrue_b.npz",
            out / "densein_b.npz",
            out / "alive_c.npz",
            out / "densein_c.npz",
            out / "ftrue_c_all.fp16.npy",
        ]
        pdir = out / "percontext"
        if pdir.exists():
            stale += sorted(pdir.glob("refit_*"))
        for p in stale:
            if p.exists():
                logger.warning("[maps] recompute: removing stale %s", p.name)
                p.unlink()
    EA._headroom(args.out_root, 2 if args.smoke else 30, "p5-maps")
    production = _production(args)
    a_dir = _assemble_dir(args)
    assert (a_dir / "split_meta.json").exists(), "maps needs the P1 outputs — run assemble"
    assert (_recapture_dir(args) / "vbar_store.npz").exists(), (
        "maps needs the P2 recapture store — run recapture"
    )
    assert (_sae_out_dir(args) / "sae_weights.safetensors").exists(), (
        "maps needs the P4 SAE-c weights — run sae_train"
    )
    scratch = _ea_scratch(args, production)
    _run_ea_refits(args, out, scratch, resume_ok)
    g1 = _gate_g1(args, out, production)  # halts fits on production FAIL (plan §7)
    if not (out / "ib_c.npz").exists():
        _ib_arm_c(args, out, scratch)
    armb_doc = _armb_all(args, out, production)
    densec_doc = _dense_companion_c(args, out, scratch, production)
    _write_json(
        meta_path,
        {
            "g1": g1["verdict"],
            "armb": armb_doc,
            "dense_companion_c": densec_doc,
            "identity_bias_note_dense_companion": (
                "identity+bias inapplicable for the dense-input companions "
                "(d_in 3584 != d_out n_alive; stated per the mapping-baselines rule)"
            ),
            "ib_convention": "bias fit on the refit_holdout train rows (map-matched)",
            "armb_ib_convention": "bias fit on the arm-b ridge tr rows (carve-matched)",
        },
        phase="maps",
    )
    _sentinel("maps", f"P5 done (g1={g1['verdict']})")
    logger.info("[maps] done (g1=%s)", g1["verdict"])


# ── P6: encode + DVs + stats (all vectorized) ────────────────────────────────────


def _r2_only(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Batched per-feature held-out R2 (fp64; no rank stats — the cheap kernel
    for corpus splits + shuffle nulls)."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    mu = t.mean(0)
    ss_res = ((t - p) ** 2).sum(0)
    ss_tot = ((t - mu) ** 2).sum(0)
    return np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


def _shuffle_null_r2(pred: np.ndarray, true: np.ndarray, seeds) -> np.ndarray:
    """K row-shuffle null draws of the per-feature R2 (advisory floor, plan §6):
    the SAME vectorized R2 kernel as the observed read, prediction rows permuted."""
    t = np.asarray(true, np.float64)
    mu = t.mean(0)
    ss_tot = ((t - mu) ** 2).sum(0)
    out = np.zeros((len(seeds), t.shape[1]), np.float16)
    n = t.shape[0]
    for i, seed in enumerate(seeds):
        perm = np.random.default_rng(seed).permutation(n)
        ss_res = ((t - np.asarray(pred[perm], np.float64)) ** 2).sum(0)
        r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
        out[i] = r2.astype(np.float16)
        print(f"[eval] shuffle-null draw {i + 1}/{len(seeds)}", flush=True)
    return out


def _rho_ceiling(tier: np.ndarray) -> float:
    """Max achievable |Spearman(tier, y)| given tier's tie structure (plan §6
    reporting duty): Pearson of tier midranks vs a perfectly tier-monotone y."""
    n = len(tier)
    if n < 3:
        return float("nan")
    order = np.argsort(tier, kind="stable")
    y = np.empty(n, np.float64)
    y[order] = np.arange(1, n + 1, dtype=np.float64)
    rt = EA._midrank(np.asarray(tier, np.float64)[:, None])[:, 0]
    a = rt - rt.mean()
    b = y - y.mean()
    den = float(np.sqrt((a**2).sum() * (b**2).sum()))
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def _wilson(p_hat: float, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson 95% CI on a proportion (acc@k reporting, plan §6)."""
    if n <= 0:
        return None
    den = 1.0 + z * z / n
    c = p_hat + z * z / (2 * n)
    h = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))
    return [float((c - h) / den), float((c + h) / den)]


def _boot_median_ci(v: np.ndarray, n_boot: int, rng, chunk: int = 1000) -> list[float] | None:
    """Batched bootstrap 95% CI on median(v) (index-matrix gather per block)."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return None
    qs = []
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        qs.append(np.median(v[rng.integers(0, len(v), (k, len(v)))], axis=1))
    d = np.concatenate(qs)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def _boot_paired_median_diff(a, b, n_boot: int, rng, chunk: int = 1000) -> dict | None:
    """Paired feature-bootstrap on median(a) − median(b) (map vs identity+bias,
    same features resampled together — plan §6)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return None
    ds = []
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        idx = rng.integers(0, len(a), (k, len(a)))
        ds.append(np.median(a[idx], axis=1) - np.median(b[idx], axis=1))
    d = np.concatenate(ds)
    return {
        "point": float(np.median(a) - np.median(b)),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "n": int(len(a)),
    }


def _spearman_pair(a: np.ndarray, b: np.ndarray) -> float | None:
    """Midrank Spearman between two per-feature vectors (bridge correlations)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return None
    ra = EA._midrank(a[ok][:, None])[:, 0]
    rb = EA._midrank(b[ok][:, None])[:, 0]
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else None


def _write_perfeature(path: Path, *, feat_ids, pred, true, counts_sel, te_prov, n_fit, floor):
    """One per-feature npz in the m-round key convention (feat_ids/r2/spearman/
    ss_tot/activity/r2_lmsys/r2_wildchat/tier) + alive-mask provenance scalars."""
    pf = EA._per_feature_metrics(pred, true)
    corpus = {}
    for label, code in (("lmsys", 0), ("wildchat", 1)):
        m = np.asarray(te_prov) == code
        if int(m.sum()) >= 2:
            corpus[label] = _r2_only(pred[m], true[m])
        else:
            corpus[label] = np.full(pred.shape[1], np.nan)
    tmp = path.parent / f".tmp_{path.name}"
    np.savez(
        tmp,
        feat_ids=np.asarray(feat_ids, np.int64),
        r2=pf["r2"],
        spearman=pf["spearman"],
        ss_tot=pf["ss_tot"],
        activity=np.asarray(counts_sel, np.float64),
        r2_lmsys=corpus["lmsys"],
        r2_wildchat=corpus["wildchat"],
        tier=S.tier_of(feat_ids),
        n_fit_rows=np.int64(n_fit),
        alive_floor=np.int64(floor),
    )
    tmp.replace(path)
    return pf


def _median_of(v: np.ndarray) -> float:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def _retrieval_cells(f_true, preds: dict, tier: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Per-tier kNN retrieval (every k x metric x predictor cell reported
    separately, no best-of — plan §6). Pool = the held-out true feature rows."""
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    out: dict = {}
    for t in (0, 1, 2):
        m = tier == t
        if int(m.sum()) == 0:
            out[str(t)] = {"note": "0 alive features in tier"}
            continue
        ft = np.ascontiguousarray(np.asarray(f_true[:, m], np.float32))
        cell: dict = {}
        for pname, parr in preds.items():
            pa = np.ascontiguousarray(np.asarray(parr[:, m], np.float32))
            pc = {}
            for metric in ("euclidean", "cosine"):
                r = knn_retrieval(pa, ft, ks=ks, metric=metric)
                r["wilson_ci_acc"] = {str(k): _wilson(r["acc_at_k"][k], r["n"]) for k in ks}
                pc[metric] = r
            cell[pname] = pc
        out[str(t)] = cell
    return out


def _tier_stats(r2_map, r2_ib, tier, activity, n_perm, n_boot, rng) -> dict:
    """Per-tier medians + bootstrap CIs, tier-diff (t0 − t2, the registered
    sign-stable difference), within-activity-quintile tier permutation (m-round
    kernel), lattice verdict, tie profile, |rho| ceiling."""
    finite = np.isfinite(r2_map)
    r2f = np.asarray(r2_map, np.float64)[finite]
    tierf = np.asarray(tier, np.int64)[finite]
    actf = np.asarray(activity, np.float64)[finite]
    if len(r2f) >= 10 and len(np.unique(tierf)) >= 2:
        strata = M._strata_of(actf, 5)  # quintiles (plan §6)
        h1 = M._tier_permutation(tierf, r2f, strata, n_perm, rng)
        h1["strata"] = "quintile"
        h1["cell_counts"] = {
            f"{s}_{t}": int(((strata == s) & (tierf == t)).sum())
            for s in np.unique(strata)
            for t in (0, 1, 2)
        }
    else:
        h1 = {
            "verdict": "insufficient-features",
            "n_features": int(len(r2f)),
            "perm_band_2p5_97p5": [float("nan"), float("nan")],
            "observed_pooled_spearman": float("nan"),
        }
    vals, cnts = np.unique(actf, return_counts=True) if len(actf) else (np.array([]), np.array([]))
    tie_profile = {
        "n_features": int(len(actf)),
        "n_unique_activity": int(len(vals)),
        "max_tie_fraction": float(cnts.max() / max(1, len(actf))) if len(cnts) else None,
    }
    per_tier = {}
    for t in (0, 1, 2):
        m = tier == t
        vm, vi = r2_map[m], r2_ib[m]
        per_tier[str(t)] = {
            "n_alive": int(m.sum()),
            "median_r2_map": M._median_iqr(vm),
            "ci95_median_map": _boot_median_ci(vm, n_boot, rng),
            "median_r2_ib": M._median_iqr(vi),
            "ci95_median_ib": _boot_median_ci(vi, n_boot, rng),
            "map_minus_ib_paired": _boot_paired_median_diff(vm, vi, n_boot, rng),
        }
    diff = {}
    for name, arr in (("map", r2_map), ("ib", r2_ib)):
        d_point = _median_of(arr[tier == 0]) - _median_of(arr[tier == 2])
        diff[name] = {
            "point_t0_minus_t2": d_point,
            "ci95": M._boot_median_diff(arr[tier == 0], arr[tier == 2], n_boot, rng),
        }
    lo, hi = h1["perm_band_2p5_97p5"]
    obs = h1["observed_pooled_spearman"]
    d_map = diff["map"]["point_t0_minus_t2"]
    if np.isfinite(obs) and obs < lo and d_map > 0:
        lattice = "coarse-better"
    elif np.isfinite(obs) and obs > hi and d_map < 0:
        lattice = "fine-better"
    else:
        lattice = "tier-null"
    return {
        "per_tier": per_tier,
        "tier_diff_t0_minus_t2": diff,
        "permutation": h1,
        "activity_tie_profile": tie_profile,
        "lattice_verdict": lattice,
        "rho_ceiling_abs": _rho_ceiling(tierf),
    }


def _arm_battery(
    out: Path,
    tag: str,
    *,
    f_true: np.ndarray,
    pred_map: np.ndarray,
    pred_ib: np.ndarray,
    feat_ids: np.ndarray,
    counts_full: np.ndarray,
    floor: int,
    n_fit_rows: int,
    te_prov: np.ndarray,
    dense_true: np.ndarray,
    dense_preds: dict[str, np.ndarray],
    sae,
    train_mean: np.ndarray,
    extra_reads: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    extra_json: dict,
    n_perm: int,
    n_boot: int,
    rng,
) -> dict:
    """One arm's full vectorized battery -> perfeature npzs + shuffle-null npz +
    tier_tests_<tag>.json + retrieval_<tag>.json. Returns {'r2_map': ...}."""
    tier = S.tier_of(feat_ids)
    counts_sel = np.asarray(counts_full, np.int64)[feat_ids]
    t0 = time.time()
    pf_map = _write_perfeature(
        out / f"perfeature_{tag}_encodepred.npz",
        feat_ids=feat_ids,
        pred=pred_map,
        true=f_true,
        counts_sel=counts_sel,
        te_prov=te_prov,
        n_fit=n_fit_rows,
        floor=floor,
    )
    print(f"[eval] unit perfeature_{tag}_map elapsed={time.time() - t0:.0f}s", flush=True)
    pf_ib = _write_perfeature(
        out / f"perfeature_{tag}_ib.npz",
        feat_ids=feat_ids,
        pred=pred_ib,
        true=f_true,
        counts_sel=counts_sel,
        te_prov=te_prov,
        n_fit=n_fit_rows,
        floor=floor,
    )
    extra_medians: dict = {}
    for name, pred_x, true_x, prov_x in extra_reads:
        pf_x = _write_perfeature(
            out / f"perfeature_{tag}_{name}.npz",
            feat_ids=feat_ids,
            pred=pred_x,
            true=true_x,
            counts_sel=counts_sel,
            te_prov=prov_x,
            n_fit=n_fit_rows,
            floor=floor,
        )
        extra_medians[name] = {
            "n_te_rows": int(true_x.shape[0]),
            "per_tier_median_r2": {str(t): _median_of(pf_x["r2"][tier == t]) for t in (0, 1, 2)},
        }
        print(f"[eval] unit perfeature_{tag}_{name} done", flush=True)

    null_r2 = _shuffle_null_r2(pred_map, f_true, SHUFFLE_SEEDS_2476)
    tmp = out / f".tmp_shuffle_null_{tag}.npz"
    np.savez(
        tmp,
        feat_ids=feat_ids,
        tier=tier,
        r2=null_r2,
        seeds=np.asarray(SHUFFLE_SEEDS_2476, np.int64),
    )
    tmp.replace(out / f"shuffle_null_{tag}.npz")
    hi = float(np.nanpercentile(null_r2.astype(np.float64), 97.5))
    rr = pf_map["r2"][np.isfinite(pf_map["r2"])]
    shuffle_doc = {
        "p97_5": hi,
        "n_seeds": len(SHUFFLE_SEEDS_2476),
        "frac_above_map": float((rr > hi).mean()) if len(rr) else None,
        "advisory": True,
    }

    stats = _tier_stats(pf_map["r2"], pf_ib["r2"], tier, counts_sel, n_perm, n_boot, rng)
    bins = (0,) + tuple(S.MATRYOSHKA_TIER_BOUNDS)
    candidates = {str(t): int(bins[t + 1] - bins[t]) for t in (0, 1, 2)}
    doc = {
        "arm": tag,
        "n_te_rows": int(f_true.shape[0]),
        "n_features_alive": int(len(feat_ids)),
        "alive_mask_provenance": {
            "criterion": (
                "active on >= ceil(0.01 * n_fit_rows) TRUE-summary fit-side encodes "
                "(never predicted codes; true-alive features retained regardless of "
                "predicted-code degeneracy)"
            ),
            "floor": int(floor),
            "n_fit_rows": int(n_fit_rows),
            "per_tier": {
                str(t): {"selected": int((tier == t).sum()), "candidate": candidates[str(t)]}
                for t in (0, 1, 2)
            },
        },
        **stats,
        "shuffle_null": shuffle_doc,
        "extra_reads": extra_medians,
        "ci_note": "bootstrap CIs conditional on the realized score rows",
        **extra_json,
    }
    _write_json(out / f"tier_tests_{tag}.json", doc, phase="eval")

    ret_preds = {
        "map": pred_map,
        "ib": pred_ib,
        "train_mean": np.broadcast_to(
            np.asarray(train_mean, np.float32), (f_true.shape[0], len(feat_ids))
        ),
    }
    retrieval = {
        "n_pool": int(f_true.shape[0]),
        "chance_note": "pool = held-out true feature rows; chance_at_k = k / n_pool",
        "tiers": _retrieval_cells(f_true, ret_preds, tier),
    }
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    dense: dict = {}
    for name, dp in dense_preds.items():
        cell: dict = {
            "pooled_r2": float(
                PR._pooled_r2(np.asarray(dp, np.float64), np.asarray(dense_true, np.float64))
            )
        }
        for metric in ("euclidean", "cosine"):
            cell[metric] = knn_retrieval(
                np.asarray(dp, np.float32), np.asarray(dense_true, np.float32), metric=metric
            )
        dense[name] = cell
    retrieval["dense_anchor"] = dense
    recon: dict = {}
    for name, arr in {"true": dense_true, **dense_preds}.items():
        fve, l0 = _recon_fve(sae, np.asarray(arr, np.float16), np.arange(len(arr)))
        recon[name] = {"var_fve": round(fve, 4), "l0": round(l0, 2)}
    retrieval["encoder_recon_fve"] = recon
    _write_json(out / f"retrieval_{tag}.json", retrieval, phase="eval")
    print(f"[eval] unit battery_{tag} done elapsed={time.time() - t0:.0f}s", flush=True)
    return {"r2_map": pf_map["r2"], "tier": tier, "lattice": doc["lattice_verdict"]}


def _bridge_b(out: Path, feat_ids_b: np.ndarray, r2_b: np.ndarray) -> dict:
    """Arm-b bridge vs the committed token-level per-feature npz (plan §6): paired
    per-feature values + per-tier bridge correlations + intersection counts."""
    committed = (
        PROJECT_ROOT
        / "eval_results"
        / "issue_1482"
        / "matryoshka_tier"
        / "perfeature_m_lmsys_default.npz"
    )
    z = np.load(committed, allow_pickle=False)
    needed = {"feat_ids", "r2", "tier", "activity"}
    assert needed <= set(z.files), (  # the plan §10 bridge-consumer key assert
        f"committed perfeature_m_lmsys_default.npz key drift: {sorted(z.files)}"
    )
    com_ids = np.asarray(z["feat_ids"], np.int64)
    com_r2 = np.asarray(z["r2"], np.float64)
    com_act = np.asarray(z["activity"], np.float64)
    inter, ia, ic = np.intersect1d(feat_ids_b, com_ids, return_indices=True)
    r2_ours = np.asarray(r2_b, np.float64)[ia]
    r2_com = com_r2[ic]
    tier_i = S.tier_of(inter)
    per_tier = {}
    for t in (0, 1, 2):
        m = tier_i == t
        per_tier[str(t)] = {
            "n_intersection": int(m.sum()),
            "spearman": _spearman_pair(r2_ours[m], r2_com[m]),
            "median_delta_turnavg_minus_token": (
                float(np.nanmedian(r2_ours[m] - r2_com[m])) if int(m.sum()) else None
            ),
        }
    pooled = {
        "n_intersection": int(len(inter)),
        "n_ours_only": int(len(feat_ids_b) - len(inter)),
        "n_committed_only": int(len(com_ids) - len(inter)),
        "spearman": _spearman_pair(r2_ours, r2_com),
        "median_delta_turnavg_minus_token": (
            float(np.nanmedian(r2_ours - r2_com)) if len(inter) else None
        ),
    }
    tmp = out / ".tmp_bridge_b.npz"
    np.savez(
        tmp,
        feat_ids=inter,
        tier=tier_i,
        r2_turnavg=r2_ours,
        r2_token_committed=r2_com,
        activity_committed=com_act[ic],
    )
    tmp.replace(out / "bridge_b.npz")
    return {"pooled": pooled, "per_tier": per_tier}


def phase_eval(args) -> None:
    """P6: gate G3 FIRST (arm-b fitness; demotion-only) -> encode predictions ->
    the vectorized per-feature / per-tier / retrieval / permutation batteries."""
    C.phase("eval")
    out = _eval_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    regime, resume_ok = _enter_phase_regime(out, args, "eval")
    finals = [
        out / n
        for n in (
            "perfeature_c_encodepred.npz",
            "tier_tests_c.json",
            "retrieval_c.json",
            "perfeature_b_encodepred.npz",
            "bridge_b.npz",
            "retrieval_b.json",
            "tier_tests_b.json",
        )
    ]
    if resume_ok and all(p.exists() for p in finals):
        logger.info("[eval] resume: all P6 deliverables present under matching regime; skip")
        return
    if not resume_ok:
        for p in [*out.glob("*.npz"), *out.glob("*.json")]:
            if p.name != "regime.json":
                logger.warning("[eval] recompute: removing stale %s", p.name)
                p.unlink()
    EA._headroom(args.out_root, 2 if args.smoke else 15, "p6-eval")
    production = _production(args)
    n_perm = min(args.n_perm, 200) if args.smoke else args.n_perm
    n_boot = min(args.n_boot, 200) if args.smoke else args.n_boot
    maps_dir = _maps_dir(args)
    a_dir = _assemble_dir(args)
    rows_present = np.load(a_dir / "rows_present.npy")
    _row_ci, prov_u8, _pools = _load_scratch_meta(args)  # stages prov.npy if absent
    y_mm = np.load(a_dir / "Y19.fp16.npy", mmap_mode="r")

    # ── G3 FIRST (plan §7): arm-b encoder fitness — DEMOTION only, never an abort ──
    store = np.load(_recapture_dir(args) / "vbar_store.npz")
    vbar20_all = np.asarray(store["vbar20"], np.float16)
    sae_lm = S.SAELensJumpReLU.load(M.SAE_IDS["lmsys"], device=args.device, cache_dir=args.sae_dir)
    fve_b, l0_b, diag_b = sae_lm.fve_l0(torch.from_numpy(vbar20_all.astype(np.float32)))
    arm_b_demoted = bool(fve_b < M.GATE_BM_HALT)
    g3 = {
        "fve": round(float(fve_b), 4),
        "l0": round(float(l0_b), 2),
        "diag": diag_b,
        "floor": M.GATE_BM_HALT,
        "n_rows": int(vbar20_all.shape[0]),
        "arm_b_demoted": arm_b_demoted,
        "verdict": "PASS" if not arm_b_demoted else "DEMOTED-exploratory-with-caveat",
    }
    sae_pile = S.SAELensJumpReLU.load(M.SAE_IDS["pile"], device=args.device, cache_dir=args.sae_dir)
    fve_p, l0_p, diag_p = sae_pile.fve_l0(torch.from_numpy(vbar20_all.astype(np.float32)))
    g3["pile_exploratory"] = {"fve": round(float(fve_p), 4), "l0": round(float(l0_p), 2)}
    _write_json(out / "gates_p6.json", {"g3": g3}, phase="eval")
    if arm_b_demoted:
        logger.warning(
            "[eval] G3 fve=%.4f below floor %.2f: arm b DEMOTED to exploratory-with-caveat "
            "(never a run abort; arm c carries the verdict)",
            fve_b,
            M.GATE_BM_HALT,
        )

    # ── arm c ──────────────────────────────────────────────────────────────────────
    sae_c = MatryoshkaBatchTopKSAE.load_local(_sae_out_dir(args), device=args.device)
    az = np.load(maps_dir / "alive_c.npz")
    alive_c = np.asarray(az["alive_ids"], np.int64)
    counts_c = np.asarray(az["counts"], np.int64)
    floor_c, n_fit_c = int(az["floor"]), int(az["n_fit_rows"])
    train_mean_c = np.asarray(az["train_mean"], np.float64)
    ftrue_all = np.load(maps_dir / "ftrue_c_all.fp16.npy", mmap_mode="r")
    f_true_c = np.asarray(ftrue_all[n_fit_c:])
    hz = np.load(maps_dir / "percontext" / "refit_holdout__ridge__seed0.npz")
    vhat = np.asarray(hz["holdout_pred16"], np.float16)
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    dz = np.load(maps_dir / "densein_c.npz")
    assert (np.asarray(dz["rows"], np.int64) == hold_rows).all(), "densein_c row-order drift"
    ibz = np.load(maps_dir / "ib_c.npz")
    assert (np.asarray(ibz["rows"], np.int64) == hold_rows).all(), "ib_c row-order drift"
    tz = np.load(maps_dir / "percontext" / "refit_lmsys_transfer__ridge__seed0.npz")
    assert (np.asarray(tz["holdout_rows"], np.int64) == hold_rows).all(), "transfer row drift"
    assert f_true_c.shape[0] == len(hold_rows), (f_true_c.shape, len(hold_rows))
    te_prov_c = prov_u8[rows_present[hold_rows]]
    ib16 = np.asarray(ibz["pred16"], np.float16)
    f_pred_c = _encode_restricted(sae_c, vhat, np.arange(len(vhat)), alive_c)
    f_ib_c = _encode_restricted(sae_c, ib16, np.arange(len(ib16)), alive_c)
    vhat_tr = np.asarray(tz["holdout_pred16"], np.float16)
    wc = te_prov_c == 1
    extra_c: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("densein", np.asarray(dz["pred16"], np.float16), f_true_c, te_prov_c),
    ]
    if int(wc.sum()) >= 2:
        f_pred_transfer = _encode_restricted(sae_c, vhat_tr, np.where(wc)[0], alive_c)
        extra_c.append(("transfer", f_pred_transfer, f_true_c[wc], te_prov_c[wc]))
    else:
        logger.warning("[eval] corpus-transfer read skipped: <2 WildChat holdout rows (smoke)")
    dense_true_c = np.asarray(y_mm[hold_rows], np.float16)
    rng_c = np.random.default_rng(BOOT_SEED_2476)
    bat_c = _arm_battery(
        out,
        "c",
        f_true=f_true_c,
        pred_map=f_pred_c,
        pred_ib=f_ib_c,
        feat_ids=alive_c,
        counts_full=counts_c,
        floor=floor_c,
        n_fit_rows=n_fit_c,
        te_prov=te_prov_c,
        dense_true=dense_true_c,
        dense_preds={"map": vhat, "ib": ib16},
        sae=sae_c,
        train_mean=train_mean_c,
        extra_reads=extra_c,
        extra_json={
            "gate_g4": json.loads((_sae_out_dir(args) / "gates_p4.json").read_text())["g4"],
            "densein_note": (
                "dense-input companion: identity+bias inapplicable (d_in 3584 != d_out n_alive)"
            ),
        },
        n_perm=n_perm,
        n_boot=n_boot,
        rng=rng_c,
    )
    del f_pred_c, f_ib_c, f_true_c, sae_c

    # ── arm b (post-G3; demotion is a labeling, the battery still runs) ────────────
    bz = np.load(maps_dir / "armb_maps.npz")
    ab = np.load(maps_dir / "alive_b.npz")
    fz = np.load(maps_dir / "ftrue_b.npz")
    db = np.load(maps_dir / "densein_b.npz")
    alive_b = np.asarray(ab["alive_ids"], np.int64)
    counts_b = np.asarray(ab["counts"], np.int64)
    floor_b, n_fit_b = int(ab["floor"]), int(ab["n_fit_rows"])
    train_mean_b = np.asarray(ab["train_mean"], np.float64)
    row_idx_all = np.asarray(fz["row_idx"], np.int64)
    row_idx_score = np.asarray(bz["row_idx_score"], np.int64)
    te_pos = np.searchsorted(row_idx_all, row_idx_score)
    assert (row_idx_all[te_pos] == row_idx_score).all(), "armb score-row alignment drift"
    assert (np.asarray(db["rows"], np.int64) == row_idx_score).all(), "densein_b row drift"
    f_true_b = np.asarray(fz["f_true"], np.float16)[te_pos]
    f_pred_b = _encode_restricted(
        sae_lm, np.asarray(bz["pred16"], np.float16), np.arange(len(row_idx_score)), alive_b
    )
    f_ib_b = _encode_restricted(
        sae_lm, np.asarray(bz["ib_pred16"], np.float16), np.arange(len(row_idx_score)), alive_b
    )
    te_prov_b = prov_u8[row_idx_score]
    extra_b: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("densein", np.asarray(db["pred16"], np.float16), f_true_b, te_prov_b),
        (
            "inlier",
            _encode_restricted(
                sae_lm,
                np.asarray(bz["pred16_inlier"], np.float16),
                np.arange(len(row_idx_score)),
                alive_b,
            ),
            np.asarray(fz["f_true_inlier_te"], np.float16),
            te_prov_b,
        ),
    ]
    ridx_store = np.asarray(store["row_idx"], np.int64)
    te_store_pos = np.searchsorted(ridx_store, row_idx_score)
    assert (ridx_store[te_store_pos] == row_idx_score).all(), "store score-row drift"
    dense_true_b = vbar20_all[te_store_pos]
    rng_b = np.random.default_rng(BOOT_SEED_2476 + 1)
    bat_b = _arm_battery(
        out,
        "b",
        f_true=f_true_b,
        pred_map=f_pred_b,
        pred_ib=f_ib_b,
        feat_ids=alive_b,
        counts_full=counts_b,
        floor=floor_b,
        n_fit_rows=n_fit_b,
        te_prov=te_prov_b,
        dense_true=dense_true_b,
        dense_preds={
            "map": np.asarray(bz["pred16"], np.float16),
            "ib": np.asarray(bz["ib_pred16"], np.float16),
        },
        sae=sae_lm,
        train_mean=train_mean_b,
        extra_reads=extra_b,
        extra_json={"gate_g3": g3, "arm_b_demoted": arm_b_demoted},
        n_perm=n_perm,
        n_boot=n_boot,
        rng=rng_b,
    )
    bridge = _bridge_b(out, alive_b, bat_b["r2_map"])
    tests_b = json.loads((out / "tier_tests_b.json").read_text())
    tests_b["bridge"] = bridge

    # pile twin (exploratory, no gate): pile-dict alive mask + encode-pred read
    fit_ids_b = np.asarray(bz["row_idx_fit"], np.int64)
    fit_store_pos = np.searchsorted(ridx_store, fit_ids_b)
    assert (ridx_store[fit_store_pos] == fit_ids_b).all(), "store fit-row drift"
    counts_pile = _encode_counts(sae_pile, vbar20_all, fit_store_pos)
    alive_pile = np.where(counts_pile >= floor_b)[0].astype(np.int64)
    if len(alive_pile) >= 1:
        ft_pile = _encode_restricted(sae_pile, vbar20_all, te_store_pos, alive_pile)
        fp_pile = _encode_restricted(
            sae_pile,
            np.asarray(bz["pred16"], np.float16),
            np.arange(len(row_idx_score)),
            alive_pile,
        )
        pf_pile = _write_perfeature(
            out / "perfeature_b_pile.npz",
            feat_ids=alive_pile,
            pred=fp_pile,
            true=ft_pile,
            counts_sel=counts_pile[alive_pile],
            te_prov=te_prov_b,
            n_fit=n_fit_b,
            floor=floor_b,
        )
        tier_pile = S.tier_of(alive_pile)
        tests_b["pile_exploratory"] = {
            "n_alive": int(len(alive_pile)),
            "per_tier_median_r2": {
                str(t): _median_of(pf_pile["r2"][tier_pile == t]) for t in (0, 1, 2)
            },
        }
    else:
        tests_b["pile_exploratory"] = {"n_alive": 0, "note": "no alive pile features"}
    _write_json(out / "tier_tests_b.json", tests_b, phase="eval")
    del sae_lm, sae_pile

    _sentinel(
        "eval",
        f"P6 done (c lattice={bat_c['lattice']}, b lattice={bat_b['lattice']}, g3={g3['verdict']})",
    )
    logger.info(
        "[eval] done: c=%s b=%s g3=%s production=%s",
        bat_c["lattice"],
        bat_b["lattice"],
        g3["verdict"],
        production,
    )


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
        "--sae-steps", type=int, default=0, help="P4: cap optimizer steps (0 = full; P0 pilot=200)"
    )
    ap.add_argument("--n-perm", type=int, default=10_000, help="P6 tier-permutation draws")
    ap.add_argument("--n-boot", type=int, default=10_000, help="P6 feature-bootstrap draws")
    ap.add_argument("--fit-n", type=int, default=0, help="EA refit train subsample (0 = all rows)")
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

        # unit-2 deferred imports (P4 save/load + P5/P6 baselines)
        from safetensors.torch import load_file, save_file  # noqa: F401

        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
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
