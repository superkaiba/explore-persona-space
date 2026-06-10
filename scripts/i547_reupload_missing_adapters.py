"""Re-upload issue-547 adapter dirs whose training-phase HF upload silently failed.

The i547 train phase uploads each cell's adapter via ``train_lora``'s
fail-soft HF path (warn + keep local copy on failure). On 2026-06-10,
32/180 uploads failed (rate-limit class burst from 4 parallel GPU shards),
which 404'd the crosseval pre-download. This script diffs the local
``adapters/i547_*`` dirs against the HF model repo and re-uploads any dir
missing ``adapter_model.safetensors``, with bounded exponential-backoff
retries. Idempotent; safe to re-run. Exits non-zero if any adapter is
still missing after the final verify pass.
"""

import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = "superkaiba1/explore-persona-space"
LOCAL = Path("/workspace/explore-persona-space/adapters")
RETRIES = 5


def _hf_complete_set(api) -> set[str]:
    """Adapter dir names on HF that already have adapter_model.safetensors."""
    files = api.list_repo_files(REPO)
    pat = re.compile(r"adapters/(i547_[^/]+)/adapter_model\.safetensors$")
    return {m.group(1) for f in files if (m := pat.match(f))}


def main() -> None:
    load_dotenv()
    from huggingface_hub import HfApi

    api = HfApi()
    on_hf = _hf_complete_set(api)
    local = sorted(
        d.name for d in LOCAL.glob("i547_*") if (d / "adapter_model.safetensors").exists()
    )
    missing = [n for n in local if n not in on_hf]
    print(f"local={len(local)} on_hf={len(on_hf)} missing={len(missing)}", flush=True)

    failed: list[str] = []
    for i, name in enumerate(missing):
        for attempt in range(RETRIES):
            try:
                api.upload_folder(
                    folder_path=str(LOCAL / name),
                    repo_id=REPO,
                    path_in_repo=f"adapters/{name}",
                )
                print(f"[{i + 1}/{len(missing)}] uploaded {name}", flush=True)
                break
            except Exception as e:
                wait = 30 * (2**attempt)
                print(
                    f"[{i + 1}/{len(missing)}] {name} attempt {attempt + 1}/{RETRIES} "
                    f"failed: {e}; sleeping {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        else:
            failed.append(name)

    still = [n for n in local if n not in _hf_complete_set(api)]
    print(f"verify: still_missing={still} upload_failed={failed}", flush=True)
    sys.exit(1 if (still or failed) else 0)


if __name__ == "__main__":
    main()
