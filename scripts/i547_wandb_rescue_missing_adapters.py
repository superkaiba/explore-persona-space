"""Rescue issue-547 adapters blocked from HF by the account storage quota.

32/180 i547 adapter uploads failed with HF 403 "public storage space
exceeded" (account-level quota — persistent, not transient). This script
gives those adapters a permanent URL WITHOUT touching the HF quota: it
diffs local ``adapters/i547_*`` dirs against the HF model repo and bundles
every dir missing ``adapter_model.safetensors`` on HF into ONE WandB
Artifact (``i547-missing-adapters``, type=model), one subdir per cell.

This is the project's established checkpoint-loss fallback (WandB
Artifacts before local deletion); the HF gap remains a Reproducibility
caveat until the quota is remediated, after which the artifact can be
downloaded and re-uploaded to HF. Idempotent: re-running logs a new
artifact version; safe.
"""

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = "superkaiba1/explore-persona-space"
LOCAL = Path("/workspace/explore-persona-space/adapters")
WANDB_PROJECT = "explore-persona-space"
ARTIFACT_NAME = "i547-missing-adapters"


def main() -> None:
    load_dotenv()
    import wandb
    from huggingface_hub import HfApi

    api = HfApi()
    pat = re.compile(r"adapters/(i547_[^/]+)/adapter_model\.safetensors$")
    on_hf = {m.group(1) for f in api.list_repo_files(REPO) if (m := pat.match(f))}
    local = sorted(
        d.name for d in LOCAL.glob("i547_*") if (d / "adapter_model.safetensors").exists()
    )
    missing = [n for n in local if n not in on_hf]
    print(f"local={len(local)} on_hf={len(on_hf)} missing={len(missing)}", flush=True)
    if not missing:
        print("nothing to rescue", flush=True)
        return

    run = wandb.init(
        project=WANDB_PROJECT,
        name="i547-adapter-rescue",
        job_type="checkpoint-rescue",
        config={"task": 547, "n_adapters": len(missing), "reason": "hf-403-storage-quota"},
    )
    art = wandb.Artifact(
        ARTIFACT_NAME,
        type="model",
        description=(
            "32 i547 LoRA adapters (role arm) whose HF upload failed with 403 "
            "public-storage-quota; permanent copy pending HF quota remediation. "
            f"Cells: {', '.join(missing)}"
        ),
    )
    for name in missing:
        art.add_dir(str(LOCAL / name), name=name)
    run.log_artifact(art)
    art.wait()
    print(f"artifact logged: {art.qualified_name} ({art.size / 1e9:.2f} GB)", flush=True)
    run.finish()
    sys.exit(0)


if __name__ == "__main__":
    main()
