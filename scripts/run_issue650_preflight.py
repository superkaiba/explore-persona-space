"""Issue #650 pinned-input preflight (runs BEFORE the pipeline's GPU phases).

Forked from ``scripts/run_issue621_preflight.py`` (origin/issue-621 @
766f44c4). Prefetches + sha-pins the reused inputs (fitness check (f),
incident #600 — resolution alone does not prove mirror identity):

- The #612 audited 60-claim false-claim pool (``eval_60.jsonl``) — pinned to
  ``EXPECTED_SYCO_CLAIM_POOL_SHA256``; staged to
  ``eval_results/issue_650/inputs/eval_60.jsonl``.
- The 3 reused #621 police_officer marker training mixes — pinned to
  ``EXPECTED_MARKER_MIX_SHA256`` (the experimenter populates the table at
  prefetch from the pinned revision; a missing pin is a LOUD KeyError, never
  a silent skip).
- Persona-registry resolution (every #650 persona resolves to a non-empty
  system prompt).

Marker token assert is in-process inside the training dispatcher
(``run_issue650_train.py::_train_marker_cell``) per the marker-leakage rule.
"""

# math/scientific notation in docstrings

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_650 import (  # noqa: E402
    EXPECTED_MARKER_MIX_SHA256,
    EXPECTED_SYCO_CLAIM_POOL_SHA256,
    HF_DATA_REPO,
    HF_MARKER_MIX_PREFIX,
    HF_MARKER_MIX_REVISION,
    HF_SYCO_CLAIM_POOL,
    SEEDS,
    SOURCE,
)
from explore_persona_space.experiments.issue_650.persona_registry import (  # noqa: E402
    assert_registry_resolves,
    load_persona_bank,
)

log = logging.getLogger("issue_650.preflight")


def _sha_assert(blob: bytes, expected: str, label: str) -> None:
    got = hashlib.sha256(blob).hexdigest()
    if got != expected:
        raise AssertionError(
            f"{label} mirror drift: sha256={got} != pinned {expected}. The HF "
            "mirror does not match the planning-time verified copy (incident "
            "#600 class). Refusing to proceed on it."
        )


def _prefetch_claim_pool() -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=HF_DATA_REPO, repo_type="dataset", filename=HF_SYCO_CLAIM_POOL)
    blob = Path(local).read_bytes()
    _sha_assert(blob, EXPECTED_SYCO_CLAIM_POOL_SHA256, HF_SYCO_CLAIM_POOL)
    dest = Path("eval_results/issue_650/inputs/eval_60.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    log.info("Claim pool pinned + staged -> %s (sha256 OK)", dest)
    return dest


def _prefetch_marker_mixes() -> list[Path]:
    from huggingface_hub import hf_hub_download

    out: list[Path] = []
    dest_dir = Path("eval_results/issue_650/training_mixes/marker")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        rel = f"{HF_MARKER_MIX_PREFIX}/{SOURCE}__seed{seed}.jsonl"
        local = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=rel,
            revision=HF_MARKER_MIX_REVISION,
        )
        blob = Path(local).read_bytes()
        # LOUD KeyError if the experimenter has not populated the pin table.
        _sha_assert(blob, EXPECTED_MARKER_MIX_SHA256[rel], rel)
        dest = dest_dir / f"{SOURCE}__seed{seed}.jsonl"
        dest.write_bytes(blob)
        out.append(dest)
        log.info("Marker mix pinned + staged -> %s (sha256 OK)", dest)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    log.info("[phase=preflight650] persona-registry resolution")
    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)

    log.info("[phase=preflight650] prefetch + sha-pin #612 claim pool")
    _prefetch_claim_pool()

    if not EXPECTED_MARKER_MIX_SHA256:
        raise AssertionError(
            "EXPECTED_MARKER_MIX_SHA256 is empty — the experimenter must populate "
            "the 3 police_officer marker-mix sha256 pins (fitness check (f), "
            "incident #600) from the pinned revision before launch. A missing pin "
            "is a HARD STOP, never a silent skip."
        )
    log.info("[phase=preflight650] prefetch + sha-pin #621 marker mixes")
    _prefetch_marker_mixes()

    log.info("[phase=preflight650_done] all pinned inputs staged + verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
