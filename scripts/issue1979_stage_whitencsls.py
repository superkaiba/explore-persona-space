"""#1979 inline round `whiten-csls-sweep` — stage the banked race ingredients.

Stages the battery tensors + Sigma Cholesky + per-mix anchors from the HF data
repo to the DATA DISK (never ``/``: 94% used at dispatch). Idempotent: an
already-present file of nonzero size is skipped.

Consumed by ``scripts/issue1979_whiten_csls_sweep.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1979_prefixrace"
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue1979_whitencsls")
ARMS_JSON = Path("eval_results/issue_1979/config/arms.json")


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    arms = json.loads(ARMS_JSON.read_text())["arms"]
    mixes = sorted({a["mix_arm_id"] for a in arms})
    rels = ["battery/ingredient_tensors.pt", "battery/sigma_chol.pt"]
    rels += [f"anchors/{m}/anchors.pt" for m in mixes]
    for rel in rels:
        dest = STAGE / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[stage] skip (present) {rel} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        hub.stage_hub_file(REPO, f"{HF_PREFIX}/{rel}", dest, repo_type="dataset")
        print(f"[stage] got {rel} ({dest.stat().st_size / 1e6:.1f} MB)")
    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"[stage] done: {len(rels)} files, {total / 1e6:.1f} MB under {STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
