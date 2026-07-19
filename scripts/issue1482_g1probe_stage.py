#!/usr/bin/env python
"""Issue #1482 G1-reconciliation probe helpers (launch plumbing, no science).

Two modes around the existing `issue1482_kresample.py --phase b2` capture:

* default (stage): copy the 16 HF-persisted B1 rollout chunks
  (`<hf-prefix>/raw_completions/gen_seed{43..46}_chunk{0..3}.json`) into the
  local `<out>/gen/` dir so a FRESH instance can run `--phase b2` without
  re-running B1 (`_load_gen_chunks` reads local files only).
* --upload: push the probe capture (`V.npz` + `capture_meta.json`) to the
  DISTINCT prefix `<hf-prefix>/g1probe_a100/` so the primary H100 capture at
  `<hf-prefix>/analysis_tensors/` is never overwritten.

Context: phase C's G1 gate FAILed on the H100 recapture (Spearman 0.9904 vs
stored A100-era e2; discrepancy is short-answer-concentrated, corr(rel,
n_ans) = -0.83). This probe re-runs the SAME capture code on the parent's
machine class (A100) to discriminate cross-GPU numerics from code drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
GEN_SEEDS = (43, 44, 45, 46)
N_CHUNKS = 4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-prefix", required=True, help="e.g. issue1482_kresample")
    ap.add_argument("--out", type=Path, required=True, help="the kresample work dir")
    ap.add_argument("--upload", action="store_true", help="upload V.npz + capture_meta")
    args = ap.parse_args()

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        for name in ("V.npz", "capture_meta.json"):
            src = args.out / name
            assert src.is_file(), f"missing probe artifact: {src}"
            dest = f"{args.hf_prefix}/g1probe_a100/{name}"
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dest,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                commit_message=f"issue-1482 G1 probe (A100 recapture): {name}",
            )
            print(f"[g1probe] uploaded {src} -> {dest}", flush=True)
        return 0

    from huggingface_hub import hf_hub_download

    gen_dir = args.out / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    n = 0
    for k in GEN_SEEDS:
        for j in range(N_CHUNKS):
            name = f"gen_seed{k}_chunk{j}.json"
            p = hf_hub_download(
                HF_DATA_REPO,
                f"{args.hf_prefix}/raw_completions/{name}",
                repo_type="dataset",
            )
            shutil.copy(p, gen_dir / name)
            n += 1
    assert n == len(GEN_SEEDS) * N_CHUNKS, f"staged {n} != 16 chunks"
    print(f"[g1probe] staged {n} gen chunks -> {gen_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
