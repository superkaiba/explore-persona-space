#!/usr/bin/env python
"""Issue #778 followup — stage the reused r_B + activation pools from HF.

The corrected-monitoring-8prompt-ladder amendment REUSES #778's cached per-layer
``r_B`` + the extraction activation pools (the drivers + null battery consume them).
This downloads them from the HF DATA repo into the consumer-exact local paths and
prints the resolved snapshot revision (SHA) so the sentinel can pin content
identity (artifact-reuse (f) / consistency D2).

Fetches (per trait in evil/sycophancy/hallucination):
  - ``analysis_tensors/rb/{trait}.pt`` -> ``data/issue_778/rb/{trait}.pt``
  - ``analysis_tensors/activations/{trait}_{pos,neg}.pt``
        -> ``data/issue_778/activations/{trait}_{pos,neg}.pt``

Shape asserts at load (rb == (28, 3584)) catch a wrong-generation mirror. Fail-loud
on any missing file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage reused r_B + activation pools from HF.")
    parser.add_argument("--issue", type=int, default=778)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--traits", nargs="+", default=list(lib.TRAITS))
    args = parser.parse_args()

    import torch
    from huggingface_hub import HfApi, hf_hub_download

    exp_name = f"issue{args.issue}_{args.slug}"
    out_root = Path(args.out_root)
    (out_root / "rb").mkdir(parents=True, exist_ok=True)
    (out_root / "activations").mkdir(parents=True, exist_ok=True)

    # Resolve the current main-revision SHA for provenance pinning (D2).
    api = HfApi()
    info = api.repo_info(DEFAULT_DATASET_REPO, repo_type="dataset", revision="main")
    revision = getattr(info, "sha", None)

    staged: list[str] = []
    for trait in args.traits:
        rb_remote = f"{exp_name}/analysis_tensors/rb/{trait}.pt"
        rb_local = hf_hub_download(
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            filename=rb_remote,
            revision="main",
        )
        dest = out_root / "rb" / f"{trait}.pt"
        shutil.copyfile(rb_local, dest)
        rb = torch.load(dest, weights_only=False)
        if tuple(rb.shape) != (lib.N_LAYERS, lib.HIDDEN_DIM):
            raise RuntimeError(
                f"reused r_B {trait} shape {tuple(rb.shape)} != "
                f"({lib.N_LAYERS},{lib.HIDDEN_DIM}) — wrong-generation mirror?"
            )
        staged.append(rb_remote)
        for side in ("pos", "neg"):
            act_remote = f"{exp_name}/analysis_tensors/activations/{trait}_{side}.pt"
            act_local = hf_hub_download(
                repo_id=DEFAULT_DATASET_REPO,
                repo_type="dataset",
                filename=act_remote,
                revision="main",
            )
            adest = out_root / "activations" / f"{trait}_{side}.pt"
            shutil.copyfile(act_local, adest)
            staged.append(act_remote)

    print(json.dumps({"staged": staged, "reused_revision": revision, "n": len(staged)}))


if __name__ == "__main__":
    main()
