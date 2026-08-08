"""Issue #1482 inline round — stage the DENSE L19 (context, answer) pair for the SAE-arm rows.

The #1482 SAE arm (P2/P3) banks only SAE-FEATURE stores; the dense answer vector
``v_x`` (mean-response residual at L19) and the dense context vector ``cx_last``
live ONLY in the 83 GB #779 n1M capture (``final_token_capture``, fp32, layers
[14, 19, 26]) — ``X.npy``/``Y.npy`` stayed pod-local and the pod is long gone
(``issue1482_error_analysis.phase_p4``: "X/Y stay pod-local (multi-GB,
regenerable via P0)"). Decoder-space scoring of any feature-space map needs
``v_x``, so this script regenerates exactly the 142,000 SAE-arm rows and nothing
else.

STREAM-REDUCE (never materialize the grid): one chunk is downloaded, its needed
rows are scattered into two fp32 memmaps, and the chunk is deleted — peak local
footprint is ~one chunk per worker (~43 MB), not 83 GB.

JOIN KEY IS ``ci``, NOT STREAM POSITION. ``row_ci.npy`` (staged from the #1482
``scratch_meta`` upload) maps global assembled row -> manifest context id; every
capture chunk carries its own ``ci`` list. Joining on ``ci`` is order-independent,
so a chunk-ordering assumption can never silently mis-align a row (the #1092
teacher-forced-capture alignment class).

DIGEST-ONLY: capture chunks carry raw LMSYS/WildChat ``prompts``; this script
reads ``ci``/``cx_last``/``v_x`` only and never prints or persists prompt text.

0 GPU. Network + CPU scatter only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM run)

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482_dense_targets")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
LAYER = 19
H_DIM = 3584


def _needed_rows(meta_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(global row ids, their ci) for holdout ++ sae_fit ++ sae_val, in that order."""
    idx = np.load(meta_dir / "split_indices.npz")
    rows = np.concatenate([idx["holdout"], idx["sae_fit"], idx["sae_val"]]).astype(np.int64)
    assert len(np.unique(rows)) == len(rows), "split union is not disjoint"
    row_ci = np.load(meta_dir / "row_ci.npy")
    ci = row_ci[rows]
    assert (ci >= 0).all(), "a needed row carries ci=-1 (pass_b head row)"
    assert len(np.unique(ci)) == len(ci), "ci is not unique over the needed rows"
    return rows, ci.astype(np.int64)


def _one_chunk(
    remote: str,
    scratch: Path,
    ci_to_pos: dict[int, int],
    layer_col: int,
    xmm: np.memmap,
    ymm: np.memmap,
    filled: np.ndarray,
) -> int:
    """Download one capture chunk, scatter its needed rows, delete it. Returns n rows."""
    from explore_persona_space.orchestrate import hub

    local = scratch / remote.rsplit("/", 1)[-1]
    try:
        hub.stage_hub_file(HF_DATA_REPO, remote, local, repo_type="dataset")
        b = torch.load(local, map_location="cpu", weights_only=True)
        assert list(b["layers"])[layer_col] == LAYER, (b["layers"], layer_col)
        cis = list(b["ci"])
        take = [(i, ci_to_pos[int(c)]) for i, c in enumerate(cis) if int(c) in ci_to_pos]
        if take:
            src = np.asarray([t[0] for t in take], dtype=np.int64)
            dst = np.asarray([t[1] for t in take], dtype=np.int64)
            xb = b["cx_last"][:, layer_col, :].to(torch.float32).numpy()
            yb = b["v_x"][:, layer_col, :].to(torch.float32).numpy()
            assert xb.shape[1] == H_DIM and yb.shape[1] == H_DIM, (xb.shape, yb.shape)
            xmm[dst] = xb[src]
            ymm[dst] = yb[src]
            filled[dst] = True
        return len(take)
    finally:
        if local.exists():
            local.unlink()  # stream-reduce: peak is ~one chunk per worker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="work root (meta/ already staged)")
    ap.add_argument("--workers", type=int, default=6, help="download pool width (#833 cap)")
    ap.add_argument("--max-chunks", type=int, default=0, help=">0: pilot slice")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    meta_dir = args.base / "meta" / "issue1482_error_analysis/analysis_tensors/scratch_meta"
    out_dir = args.base / "dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.base / "chunk_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    rows, ci = _needed_rows(meta_dir)
    n = len(rows)
    ci_to_pos = {int(c): i for i, c in enumerate(ci)}
    logger.info("[stage] need %d rows (%d unique ci)", n, len(ci_to_pos))

    xpath, ypath = out_dir / "X_L19.f32.mm", out_dir / "Y_L19.f32.mm"
    fpath = out_dir / "filled.npy"
    resume = xpath.exists() and ypath.exists() and fpath.exists()
    mode = "r+" if resume else "w+"
    xmm = np.memmap(xpath, dtype=np.float32, mode=mode, shape=(n, H_DIM))
    ymm = np.memmap(ypath, dtype=np.float32, mode=mode, shape=(n, H_DIM))
    filled = np.load(fpath) if resume else np.zeros(n, dtype=bool)
    logger.info("[stage] resume=%s already-filled=%d/%d", resume, int(filled.sum()), n)

    files = sorted(
        hub.list_hf_files_under_path(HfApi(), HF_DATA_REPO, CAPTURE_PREFIX, repo_type="dataset")
    )
    files = [f for f in files if f.endswith(".pt")]
    if args.max_chunks > 0:
        files = files[: args.max_chunks]
    logger.info("[stage] %d capture chunks", len(files))

    # layer column is fixed by the capture's own `layers` list; probe the first chunk.
    probe_local = scratch / "_probe.pt"
    hub.stage_hub_file(HF_DATA_REPO, files[0], probe_local, repo_type="dataset")
    layers = list(torch.load(probe_local, map_location="cpu", weights_only=True)["layers"])
    probe_local.unlink()
    layer_col = layers.index(LAYER)
    logger.info("[stage] capture layers=%s -> L%d at column %d", layers, LAYER, layer_col)

    t0, done, got = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_one_chunk, f, scratch, ci_to_pos, layer_col, xmm, ymm, filled): f
            for f in files
        }
        for fut in as_completed(futs):
            got += fut.result()
            done += 1
            if done % 100 == 0 or done == len(files):
                el = time.time() - t0
                logger.info(
                    "[stage] chunk %d/%d rows=%d filled=%d/%d elapsed=%.0fs eta=%.0fs",
                    done,
                    len(files),
                    got,
                    int(filled.sum()),
                    n,
                    el,
                    el / max(1, done) * (len(files) - done),
                )
                np.save(fpath, filled)  # checkpoint the coverage mask

    xmm.flush()
    ymm.flush()
    np.save(fpath, filled)
    miss = int((~filled).sum())
    logger.info(
        "[stage] done rows=%d filled=%d missing=%d in %.0fs",
        got,
        int(filled.sum()),
        miss,
        time.time() - t0,
    )
    if miss and args.max_chunks == 0:
        raise RuntimeError(f"[stage] {miss} of {n} needed rows never appeared in any chunk")
    (out_dir / "dense_targets_meta.json").write_text(
        json.dumps(
            {
                "n_rows": int(n),
                "row_ids": {"order": "holdout ++ sae_fit ++ sae_val"},
                "n_holdout": int(np.load(meta_dir / "split_indices.npz")["holdout"].shape[0]),
                "n_sae_fit": int(np.load(meta_dir / "split_indices.npz")["sae_fit"].shape[0]),
                "n_sae_val": int(np.load(meta_dir / "split_indices.npz")["sae_val"].shape[0]),
                "layer": LAYER,
                "layer_col": int(layer_col),
                "capture_prefix": CAPTURE_PREFIX,
                "n_chunks": len(files),
                "dtype": "float32",
                "fields": {
                    "X_L19.f32.mm": "cx_last@L19",
                    "Y_L19.f32.mm": "v_x@L19 (mean-response)",
                },
                "wall_time_s": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    np.save(out_dir / "row_ids.npy", rows)
    np.save(out_dir / "row_ci.npy", ci)
    for p in scratch.glob("*.pt"):
        p.unlink()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit before C-extension finalize (the #1689 atexit race)
