#!/usr/bin/env python
"""Stream the #1345 turnstore into a small layer-sliced cache (full n).

The four r1/r2 stores are 87 GB of 28-layer bf16 activations, which exceeds
``VM_ANALYSIS_FOOTPRINT_GB_MAX`` (50 GB) and would otherwise force the ladder
round onto the cpu-bigmem GCP lane. But the ladder only ever reads a handful of
frozen layers, so we take the rule's other branch — "stream the data without
materializing it locally":

    for each shard:  download -> load -> keep ONLY the wanted layers -> delete

Peak on-disk is ONE shard (2.24 GB) plus the growing slice cache; the finished
cache is ~2.2 GB per store (~8.7 GB total) at fp16, versus 87 GB for the raw
stores. That buys the FULL n = 4,724 matched rows, which matters: at d = 3584 a
1,877-row subset sits at n/d ~ 0.4, below the interpolation threshold, and the
within-regime ceiling collapses from the published 0.625 to 0.099 — a rung
verdict read against that ceiling would not describe the published operator.

Cache layout (one .pt per stem):
    {"conv_ids": [...], "slots": (n, 2, L, d) fp16, "profiles": (n, 2, L, d) fp16,
     "layers": [...]}
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE torch: torch freezes its intra-op thread pool from
# OMP_NUM_THREADS at import, so the shared-VM thread caps (#847) only bind
# in-process if the env is populated first.
load_dotenv()
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch  # noqa: E402

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue1345_common as c  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

HF_PREFIX = "issue1345_framing/analysis_tensors/turnstore"


def extract_stem(
    stem: str, dl_dir: Path, cache_dir: Path, layers: list[int], n_shards: int, *, keep_shards: bool
) -> Path:
    out = cache_dir / f"{stem}_L{'-'.join(map(str, layers))}.pt"
    if out.exists():
        print(f"[{stem}] cache hit {out.name}", flush=True)
        return out
    li = torch.as_tensor(layers)
    conv_ids: list[str] = []
    slots: list[torch.Tensor] = []
    profs: list[torch.Tensor] = []
    for k in range(n_shards):
        t0 = time.time()
        rel = f"{HF_PREFIX}/{stem}_shard{k:03d}.pt"
        # Routed through retry_transient per the fleet-shared HF budget rule
        # (#1547): these 40 shard pulls share the org-wide budget with every
        # other live run, so a 429 storm must self-throttle on Retry-After
        # rather than kill the extract.
        p = Path(
            hub.retry_transient(
                lambda rel=rel: hf_hub_download(
                    c.HF_DATA_REPO, rel, repo_type="dataset", local_dir=str(dl_dir)
                ),
                what=f"hf_hub_download({stem}_shard{k:03d}.pt)",
            )
        )
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = len(d["conv_ids"])
        assert len(d["slots"]) == n and len(d["profiles"]) == n, (stem, k, n)
        slots.append(torch.stack(d["slots"]).index_select(2, li).to(torch.float16))
        profs.append(torch.stack(d["profiles"]).index_select(2, li).to(torch.float16))
        conv_ids.extend(str(x) for x in d["conv_ids"])
        del d
        if not keep_shards:
            # the shard is a re-downloadable cache; the HF blob + symlink both go
            real = p.resolve()
            p.unlink(missing_ok=True)
            if real.exists():
                real.unlink(missing_ok=True)
        print(
            f"[{stem}] shard{k:03d} n={n} cum={len(conv_ids)} {time.time() - t0:.0f}s", flush=True
        )
    torch.save(
        {
            "conv_ids": conv_ids,
            "slots": torch.cat(slots),
            "profiles": torch.cat(profs),
            "layers": layers,
        },
        out,
    )
    print(
        f"[{stem}] wrote {out} rows={len(conv_ids)} {out.stat().st_size / 1e9:.2f} GB", flush=True
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=list(cm.FROZEN_LAYERS))
    ap.add_argument("--models", nargs="+", default=["instruct", "pretrained"])
    ap.add_argument("--regimes", nargs="+", default=["r1", "r2"])
    ap.add_argument("--n-shards", type=int, default=10)
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for model in args.models:
        for reg in args.regimes:
            stem = f"{model}_{c.REGIME_FORMAT[reg]}_{c.TRACK}"
            extract_stem(
                stem,
                args.dl_dir,
                args.cache_dir,
                args.layers,
                args.n_shards,
                keep_shards=args.keep_shards,
            )
    print(f"TOTAL {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
