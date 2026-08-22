"""Stream-reduce the ANSWER-span activation (v_A = `t1` pooling) for the
compliance-labelled jailbreak contexts from evil_labeling.tar, 6 layers, deduped
per context. Sibling of issue1739_jbmine_stream_evil.py (which pulls context_end
= v_C); this pulls t1 = v_A, the map's Y target, for the evil contexts (already
captured — 0-GPU). Same stream-reduce pattern; never materializes the 32 GB tar.

Content hygiene: retains activation tensors + row_index (context_id + numeric
positions) only; never reads rollout/prompt text.

Output: <DEST>/evil_answer_t1.npz with arrays
  context_ids (str, n_ctx), layers (int, 6), X (fp16, n_ctx x 6 x 3584).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.issue1739_map963k_slice import (  # noqa: E402
    ParallelRangeReader,
    head_size,
    tar_url,
    wanted_re,
)

DEST = Path(f"/mnt/eps-data/{os.environ['USER']}/issue1739_jbmine")
LAYERS = (7, 11, 15, 19, 23, 27)
HIDDEN = 3584
KIND = "t1"  # answer-span representation (v_A); same pooling the benign map used


def _shard_num(name: str) -> int:
    return int(name.split("_shard")[1].split(".")[0]) if "_shard" in name else 0


def main() -> int:
    token = os.environ["HF_TOKEN"]
    url = tar_url("evil", "main")
    total = head_size(url, token)
    want = wanted_re((KIND,), LAYERS)
    print(f"[stream] evil_labeling.tar {total / 1e9:.1f} GB; retaining {KIND} {LAYERS}", flush=True)

    ans: dict[int, dict[int, np.ndarray]] = {L: {} for L in LAYERS}
    row_ctx: dict[int, list[str]] = {}
    t0 = time.time()
    seen = 0
    reader = ParallelRangeReader(url, token=token, total=total, window=48 << 20, workers=16)
    buffered = io.BufferedReader(reader, buffer_size=48 << 20)
    with tarfile.open(fileobj=buffered, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.rsplit("/", 1)[-1]
            if not (want.match(name) or name.startswith("row_index")):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            raw = fh.read()
            seen += 1
            if name.startswith("row_index"):
                s = _shard_num(name)
                row_ctx[s] = [
                    json.loads(ln)["context_id"]
                    for ln in raw.decode("utf-8").split("\n")
                    if ln.strip()
                ]
            else:  # t1_L{L}_shard{S}.npy
                L = int(name.split("_L")[1][:2])
                s = _shard_num(name)
                ans[L][s] = np.load(io.BytesIO(raw), allow_pickle=False)
            if seen % 40 == 0:
                print(f"  [stream] {seen} members, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[stream] done: {seen} members in {(time.time() - t0) / 60:.1f} min", flush=True)

    shards = sorted(row_ctx)
    ctx_ids: list[str] = []
    for s in shards:
        ctx_ids.extend(row_ctx[s])
    n_rows = len(ctx_ids)
    Xrows = np.zeros((len(LAYERS), n_rows, HIDDEN), np.float32)
    for li, L in enumerate(LAYERS):
        off = 0
        for s in shards:
            a = ans[L][s]
            Xrows[li, off : off + a.shape[0]] = a.astype(np.float32)
            off += a.shape[0]
        assert off == n_rows, f"L{L}: {off} rows vs {n_rows} row_index"

    uniq = sorted(set(ctx_ids))
    idx_by_ctx: dict[str, list[int]] = {}
    for i, c in enumerate(ctx_ids):
        idx_by_ctx.setdefault(c, []).append(i)
    X = np.zeros((len(uniq), len(LAYERS), HIDDEN), np.float16)
    for j, c in enumerate(uniq):
        rows = idx_by_ctx[c]
        for li in range(len(LAYERS)):
            X[j, li] = Xrows[li, rows].mean(axis=0).astype(np.float16)

    out = DEST / "evil_answer_t1.npz"
    np.savez(out, context_ids=np.array(uniq), layers=np.array(LAYERS), X=X)
    print(
        f"[done] {len(uniq)} contexts from {n_rows} rows -> {out} "
        f"({X.nbytes / 1e6:.0f} MB, {(time.time() - t0) / 60:.1f} min total)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
