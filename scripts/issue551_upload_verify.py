"""Issue #551 — upload re-extracted shift tensors to the HF data repo + fail-loud verify.

Extracted verbatim from the upload/verify heredoc that previously lived inline in
scripts/run_issue551_extract.sh step 7, so the pod driver and the VM-side smoke run
the SAME code path (round-2 fix for the `upload-verify-smoke-missing` concern).

Uploads `<output_dir>/shifts` to `<repo_id>` (repo_type=dataset) under `<prefix>`
(upload_folder, one retry, per-file fallback), then verifies via the Python Hub API
`list_repo_files` (never the `hf` CLI) that exactly 18 `.pt` + 18 `.manifest.json`
landed under the prefix. Exits 2 on a count miss so the driver's fail-loud branch
(`upload_verify_failed_POD_KEPT_ALIVE`) fires.

Usage:
    uv run python scripts/issue551_upload_verify.py <output_dir> <repo_id> <prefix>
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, list_repo_files, upload_folder


def main() -> None:
    load_dotenv()

    output_dir, repo_id, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    token = os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import get_token

        token = get_token()
    assert token, "No HF token in env or HF cache — refusing a doomed upload."

    folder = output_dir / "shifts"
    try:
        upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=prefix,
            token=token,
        )
    except Exception as e:  # one retry, then per-file fallback
        print(f"upload_folder attempt 1 failed: {type(e).__name__}: {e}; retrying once")
        time.sleep(30)
        try:
            upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=prefix,
                token=token,
            )
        except Exception as e2:
            print(f"upload_folder attempt 2 failed: {type(e2).__name__}: {e2}; per-file fallback")
            api = HfApi(token=token)
            for f in sorted(folder.iterdir()):
                if f.suffix not in {".pt", ".json"}:
                    continue
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f"{prefix}/{f.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

    # Fail-loud verification via the Python Hub API (never the `hf` CLI).
    files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith(prefix + "/")]
    n_pt = sum(1 for f in files if f.endswith(".pt"))
    n_mf = sum(1 for f in files if f.endswith(".manifest.json"))
    print(f"verified on hub: {n_pt} .pt + {n_mf} .manifest.json under {prefix}/")
    if n_pt != 18 or n_mf != 18:
        print(f"UPLOAD VERIFY FAIL: expected 18 + 18, got {n_pt} + {n_mf}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
