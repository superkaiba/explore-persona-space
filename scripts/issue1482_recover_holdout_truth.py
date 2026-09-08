#!/usr/bin/env python3
"""Issue #1482: recover the layer-19 ANSWER-STATE TRUTH for the 20,000-context
main holdout, the one array the round never banked.

WHY. The feature-level analysis in the paper (Section "information", appendix
`app:sae-properties`) scores each SAE feature by the held-out R^2 of a map from
h_C to that feature's MEAN ACTIVATION -- a target that passes through the SAE
encoder and the BatchTopK gate, so it is mostly zeros for a rare feature. A
reviewer asked whether the same property ranking holds for DECODER DIRECTIONS
instead: predict d_f^T v_x, the answer state's component along feature f's unit
decoder column, which is dense, defined on every answer, and has no encoder, no
gate and no threshold. That read needs the true answer states v_x on the same
holdout the paper's numbers use.

WHAT IS AND IS NOT ALREADY BANKED (verified 2026-09-08):
  * PRESENT  full-dim predictions, (20000, 3584) fp16, ridge and MLP, both on
             local disk and HF (`percontext/refit_holdout__*__seed0.npz`,
             key `holdout_pred16`), plus per-row scalars `holdout_e2`
             (||Y - pred||^2) and `holdout_denom`, and `holdout_rows` (the
             global ci of each holdout context, in row order).
  * ABSENT   the truth Y itself. It exists only inside the parent capture
             chunks, `issue779_monitoring/fitter-fair-comparison-n1m/
             final_token_capture`: 1,920 .pt files, 83.22 GB, each a bundle
             with `v_x` (n, 3, H), `cx_last` (n, 3, H) and `ci` (n,).

WHY NOT THE PARENT'S OWN STREAMER. `N1M._stream_n1m_layer` streams the same
chunks but ACCUMULATES EVERY ROW: 964,844 x 3584 fp32 for cx_last AND v_x is
~27.7 GB resident, to keep the 143 MB we actually want. This module reuses the
parent's per-chunk primitives verbatim -- the same scoped+retried chunk listing,
the same `_download_chunk_with_retry`, the same `F._mmap_load` and
`N50._slice_layer` -- and applies the holdout row filter INSIDE the loop, so
peak RAM is the output array and peak disk is one chunk (~43 MB). Download,
slice, keep ~10 rows, delete, next.

THE GATE, and why it is conclusive. `holdout_e2[i]` is the banked per-row
squared error ||Y[i] - pred[i]||^2 of the ridge refit. We hold pred. Once Y is
recovered we recompute that quantity and compare. It needs no centering
convention and no assumption about the fitting pipeline, so agreement proves
the recovered rows ARE the array the paper's numbers were computed from, in the
right order. A mismatch fails loud rather than persisting a plausible-looking
wrong matrix.

0 GPU. Network-bound: 83 GB of transfer, which is why this runs on a CPU pod
and not the shared VM (CLAUDE.md standing rule: anything downloading a lot of
data runs on a pod even when it is CPU-only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_common as C  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

LAYER = 19
CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
PRED_REPO_PATH = (
    "issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz"
)
OUT_PREFIX = "issue1482_error_analysis/analysis_tensors/holdout_truth"
N_HOLDOUT = 20000
# The #1482 driver assembles Y as concat([pass_b_block, streamed_n1m_rows]), so a
# `holdout_rows` value is an index into THAT matrix, not a capture ci. The first
# N_PASS_B rows are the parent's pass-B bundle; a streamed row at capture index
# ci therefore sits at assembled index ci + N_PASS_B. Selecting on raw ci instead
# silently returns real-but-wrong rows (the identity gate caught exactly that).
# And the join key is STREAM POSITION, not ci: 156 of the 960,000 manifest
# contexts never captured, so ci skips values and drifts away from position
# after the first gap. np.concatenate preserves stream ORDER, so position is
# what indexes the assembled matrix. ci is kept for provenance only.
N_PASS_B = N1M.N_PASS_B  # 5000
H = 3584
GATE_RTOL = 2e-3  # fp16 storage round-trip on a ~10^2-scale squared norm


def _log(msg: str) -> None:
    print(f"[recover-truth {datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pred_bundle(local_pred: Path | None, cache_dir: Path) -> dict:
    """holdout_rows + holdout_pred16 + holdout_e2, from local disk or HF."""
    if local_pred is not None and local_pred.exists():
        _log(f"prediction bundle (local): {local_pred}")
        path = local_pred
    else:
        from huggingface_hub import hf_hub_download

        _log(f"prediction bundle (HF): {PRED_REPO_PATH}")
        path = Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    C.HF_DATA_REPO,
                    PRED_REPO_PATH,
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                ),
                what="holdout prediction bundle",
            )
        )
    with np.load(path) as z:
        rows = np.asarray(z["holdout_rows"], dtype=np.int64)
        pred = np.asarray(z["holdout_pred16"])
        e2 = np.asarray(z["holdout_e2"], dtype=np.float64)
    assert rows.shape == (N_HOLDOUT,), rows.shape
    assert pred.shape == (N_HOLDOUT, H), pred.shape
    assert len(set(rows.tolist())) == N_HOLDOUT, "holdout ci are not unique"
    return {"rows": rows, "pred": pred, "e2": e2}


def stream_holdout_rows(
    want: np.ndarray,
    cache_dir: Path,
    max_chunks: int,
    revision: str | None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse of the parent chunk loop with a row filter. Returns (v16, ci)."""
    from huggingface_hub import HfApi

    names = hub.retry_transient(
        lambda: sorted(
            f.path.rsplit("/", 1)[-1]
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right here
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO,
                path_in_repo=CAPTURE_PREFIX,
                repo_type="dataset",
                recursive=True,
                revision=revision,
            )
            if getattr(f, "size", None) is not None and f.path.endswith(".pt")
        ),
        what=f"n1m chunk listing ({CAPTURE_PREFIX})",
    )
    if not names:
        raise FileNotFoundError(f"no capture chunks under HF {CAPTURE_PREFIX}")
    # Selection keys on STREAM POSITION, which is only defined by walking chunks
    # in order from the first, so striding is refused rather than silently
    # mis-joining. A smoke takes a contiguous PREFIX (--max-chunks); the first
    # holdout row sits at position 176, inside chunk 0, so a short prefix
    # exercises the real join.
    if stride > 1:
        raise RuntimeError(
            "--chunk-stride is unsupported: stream position requires contiguous chunks. "
            "Use --max-chunks for a contiguous-prefix smoke."
        )
    if max_chunks:
        names = names[:max_chunks]
    _log(f"{len(names)} capture chunks to stream")

    if int(want.min()) < N_PASS_B:
        raise RuntimeError(
            f"holdout index {int(want.min())} falls inside the {N_PASS_B}-row pass-B block, "
            "which this streamer does not cover"
        )
    wanted = set(int(x) - N_PASS_B for x in want.tolist())  # stream positions
    off = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    got_v: list[np.ndarray] = []
    got_pos: list[int] = []
    got_ci: list[int] = []
    t0 = time.time()

    for i, name in enumerate(names):
        path = Path(
            N1M._download_chunk_with_retry(
                C.HF_DATA_REPO, f"{CAPTURE_PREFIX}/{name}", cache_dir, revision
            )
        )
        b = N1M.F._mmap_load(path)
        ci = np.asarray([int(x) for x in b["ci"]], dtype=np.int64)
        sel = np.array([j for j in range(int(ci.size)) if (off + j) in wanted], dtype=np.int64)
        ci_lo, ci_hi = (int(ci.min()), int(ci.max())) if ci.size else (-1, -1)
        if sel.size:
            vx = N50._slice_layer(b, "v_x", LAYER)  # (n, H)
            got_v.append(np.asarray(vx[sel], dtype=np.float16))
            got_pos.extend((off + sel).tolist())
            got_ci.extend(ci[sel].tolist())
        off += int(ci.size)
        del b
        path.unlink()
        if (i + 1) % 100 == 0 or (i + 1) == len(names):
            el = time.time() - t0
            _log(
                f"chunk {i + 1}/{len(names)} ci[{ci_lo},{ci_hi}]  rows kept {len(got_ci)}  "
                f"{el / (i + 1):.2f}s/chunk  eta {(len(names) - i - 1) * el / (i + 1) / 60:.1f}min"
            )

    if not got_v:
        raise RuntimeError("no holdout rows found in any chunk -- wrong prefix or row convention")
    return (
        np.concatenate(got_v),
        np.asarray(got_pos, dtype=np.int64),
        np.asarray(got_ci, dtype=np.int64),
    )


def reorder_to_holdout(v16: np.ndarray, pos: np.ndarray, rows: np.ndarray, partial: bool):
    """Put recovered rows into holdout row order. Fails loud on a gap unless partial."""
    at = {int(c): j for j, c in enumerate(pos.tolist())}
    missing = [int(r) for r in rows.tolist() if int(r) - N_PASS_B not in at]
    if missing and not partial:
        raise RuntimeError(
            f"{len(missing)} of {len(rows)} holdout contexts absent from the stream "
            f"(first few: {missing[:5]})"
        )
    keep = np.array(
        [j for j, r in enumerate(rows.tolist()) if int(r) - N_PASS_B in at], dtype=np.int64
    )
    order = np.array([at[int(rows[j]) - N_PASS_B] for j in keep.tolist()], dtype=np.int64)
    return v16[order], keep


def run_gate(v16: np.ndarray, pred: np.ndarray, e2: np.ndarray, keep: np.ndarray) -> dict:
    """Recompute ||Y - pred||^2 and compare with the banked per-row value."""
    y = np.asarray(v16, dtype=np.float64)
    p = np.asarray(pred[keep], dtype=np.float64)
    got = ((y - p) ** 2).sum(axis=1)
    ref = e2[keep]
    denom = np.maximum(np.abs(ref), 1e-12)
    rel = np.abs(got - ref) / denom
    out = {
        "n_rows_gated": int(keep.size),
        "max_rel_err": float(rel.max()),
        "median_rel_err": float(np.median(rel)),
        "frac_within_rtol": float((rel <= GATE_RTOL).mean()),
        "rtol": GATE_RTOL,
        "banked_e2_median": float(np.median(ref)),
        "recomputed_e2_median": float(np.median(got)),
    }
    out["verdict"] = "PASS" if out["max_rel_err"] <= GATE_RTOL else "FAIL"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/workspace/out/issue1482_holdout_truth")
    ap.add_argument("--cache-dir", default="/workspace/scratch/i1482_chunks")
    ap.add_argument("--local-pred", default=None, help="local refit_holdout__ridge__seed0.npz")
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke: stream only N chunks")
    ap.add_argument(
        "--chunk-stride",
        type=int,
        default=1,
        help="smoke: take every Nth chunk so the sample spans the whole corpus",
    )
    ap.add_argument("--revision", default=None)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    smoke = bool(args.max_chunks) or args.chunk_stride > 1

    bundle = load_pred_bundle(Path(args.local_pred) if args.local_pred else None, cache_dir)
    v_raw, pos_raw, ci_raw = stream_holdout_rows(
        bundle["rows"], cache_dir, args.max_chunks, args.revision, args.chunk_stride
    )
    _log(f"recovered {pos_raw.size} rows from the stream")

    v16, keep = reorder_to_holdout(v_raw, pos_raw, bundle["rows"], partial=smoke)
    gate = run_gate(v16, bundle["pred"], bundle["e2"], keep)
    _log(f"identity gate: {json.dumps(gate)}")
    if gate["verdict"] != "PASS":
        raise SystemExit(
            f"IDENTITY GATE FAIL: max relative error {gate['max_rel_err']:.3e} exceeds "
            f"{GATE_RTOL:.1e}. The recovered rows are NOT the array the banked "
            f"per-row errors came from. Nothing persisted."
        )

    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "layer": LAYER,
        "capture_prefix": CAPTURE_PREFIX,
        "revision": args.revision,
        "n_rows": int(v16.shape[0]),
        "smoke": smoke,
        "max_chunks": args.max_chunks,
        "chunk_stride": args.chunk_stride,
        "gate": gate,
        "smoke_blind_spots": (
            "none -- the smoke path runs the same download/slice/filter/gate code as "
            "production; it differs only in HOW MANY chunks it walks, so it certifies "
            "the ci convention, the layer slice and the gate arithmetic, and does NOT "
            "certify that every holdout context is present (partial=True skips the "
            "completeness check that production enforces)."
        ),
    }
    tag = "smoke" if smoke else "full"
    npz_path = out_dir / f"v_holdout_L{LAYER}_{tag}.npz"
    np.savez_compressed(npz_path, v16=v16, ci=bundle["rows"][keep].astype(np.int64), keep=keep)
    (out_dir / f"recover_meta_{tag}.json").write_text(json.dumps(meta, indent=1))
    _log(f"wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    if args.upload and not smoke:
        from huggingface_hub import HfApi

        api = HfApi()
        for p in sorted(out_dir.iterdir()):
            hub.retry_transient(
                lambda p=p: api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"{OUT_PREFIX}/{p.name}",
                    repo_id=C.HF_DATA_REPO,
                    repo_type="dataset",
                ),
                what=f"upload {p.name}",
            )
            _log(f"uploaded {OUT_PREFIX}/{p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
