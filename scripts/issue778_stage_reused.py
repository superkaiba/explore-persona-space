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

Content-identity pins (plan v5 §12(f) / artifact-reuse (f), reconciler round 1):
  - The reused ``r_B`` files are content-pinned against ``EXPECTED_SHA256`` — the
    local sha256 is asserted == the recorded value BEFORE the shape assert, so a
    silently re-uploaded / wrong-generation mirror of the same shape fails loud.
  - The download ``revision`` is pinned to the CURRENT resolved commit SHA (not
    mutable ``"main"``) so every file in the fetch comes from the same immutable
    snapshot; the resolved SHA is printed for the sentinel.
  - The 6 activation pools have no plan-recorded sha256; their resolved LFS
    sha256 (from ``repo_info(files_metadata=True)`` at the pinned revision) is
    CAPTURED into the printed staged-manifest (pin-at-fetch) so a later re-run
    can verify content identity against this run's recorded values.

Shape asserts at load (rb == (28, 3584)) stay as the SECOND check. Fail-loud on
any missing file, sha256 mismatch, or shape mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO

load_dotenv()

# Plan v5 §12(f) content-identity pins for the reused per-layer r_B tensors
# (the (28, 3584) diff-of-means directions produced by parent #778's extraction).
# A silently re-uploaded / wrong-generation mirror of the same shape passes the
# shape assert but fails this sha256 assert (the K3 recipe gate is a downstream
# loud backstop, but a subtly-wrong-but-plausible mirror could clear it — this pin
# is what catches that class). Regression: tests/test_issue778_stage_reused.py.
EXPECTED_SHA256: dict[str, str] = {
    "evil": "67d1caafe536f11de29367b48a59f3c6bd372d01a6c44f46a82c6203b1c5ebdb",
    "hallucination": "8bea89cd0e2f43eb902d0fcff544a3eed2fc4006ec79b3bd440b785852db4a6f",
    "sycophancy": "20e498a2a3aca5450c731ac031cc13d887080a432b355e84055bc664d6087ec5",
}


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 digest of ``path`` (streamed, 1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_rb_sha256(trait: str, dest: Path) -> str:
    """Fail-loud unless ``dest``'s sha256 == the plan-recorded value for ``trait``.

    Returns the verified digest. Raises RuntimeError on an unknown trait (no pin
    recorded) or a mismatch (wrong-generation mirror).
    """
    if trait not in EXPECTED_SHA256:
        raise RuntimeError(
            f"reused r_B {trait!r} has no EXPECTED_SHA256 pin — refuse to stage an "
            f"unpinned reused artifact (plan §12(f))"
        )
    got = _sha256_file(dest)
    if got != EXPECTED_SHA256[trait]:
        raise RuntimeError(
            f"reused r_B {trait} sha256 {got} != expected {EXPECTED_SHA256[trait]} "
            f"— wrong-generation / re-uploaded mirror (content-identity pin, §12(f))"
        )
    return got


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

    # Resolve the current main-revision SHA ONCE and pin EVERY fetch to it, so all
    # staged files come from the SAME immutable snapshot (never mutable "main",
    # which can drift between the resolve and the per-file downloads). D2 / §12(f).
    api = HfApi()
    info = api.repo_info(DEFAULT_DATASET_REPO, repo_type="dataset", revision="main")
    revision = getattr(info, "sha", None)
    if not revision:
        raise RuntimeError(
            f"could not resolve {DEFAULT_DATASET_REPO} main revision SHA — refuse to "
            f"fetch at mutable 'main' (§12(f) revision pin)"
        )

    # Map remote path -> resolved LFS sha256 (pin-at-fetch for the activation pools,
    # which have no plan-recorded sha256). repo_info(files_metadata=True) returns
    # per-sibling lfs.sha256 for LFS files at the pinned revision.
    lfs_sha: dict[str, str] = {}
    info_meta = api.repo_info(
        DEFAULT_DATASET_REPO, repo_type="dataset", revision=revision, files_metadata=True
    )
    for sib in getattr(info_meta, "siblings", None) or []:
        lfs = getattr(sib, "lfs", None)
        if lfs is not None:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if sha:
                lfs_sha[sib.rfilename] = sha

    staged: list[str] = []
    rb_sha256: dict[str, str] = {}
    activation_sha256: dict[str, str] = {}
    for trait in args.traits:
        rb_remote = f"{exp_name}/analysis_tensors/rb/{trait}.pt"
        rb_local = hf_hub_download(
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            filename=rb_remote,
            revision=revision,
        )
        dest = out_root / "rb" / f"{trait}.pt"
        shutil.copyfile(rb_local, dest)
        # FIRST check: content-identity sha256 pin (plan §12(f)). A wrong-generation
        # mirror of the same shape passes the shape assert but fails this.
        rb_sha256[trait] = assert_rb_sha256(trait, dest)
        # SECOND check: shape.
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
                revision=revision,
            )
            adest = out_root / "activations" / f"{trait}_{side}.pt"
            shutil.copyfile(act_local, adest)
            staged.append(act_remote)
            # Pin-at-fetch: record the resolved LFS sha256 (Hub-side) into the manifest.
            if act_remote in lfs_sha:
                activation_sha256[act_remote] = lfs_sha[act_remote]

    print(
        json.dumps(
            {
                "staged": staged,
                "reused_revision": revision,
                "n": len(staged),
                "rb_sha256": rb_sha256,
                "activation_sha256": activation_sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
