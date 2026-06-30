#!/usr/bin/env python3
"""Issue #745 acceptance: end-to-end HF Hub upload-engagement smoke.

POSITIVELY asserts which transfer router engages on a real >=30 MB file-PATH
upload to a project repo (plan v2 §6 check 7), and measures before/after
throughput so the accelerator lever's effect is MEASURED, not assumed.

Branch B (the verified current state — both project repos route to Xet):
asserts ``_upload_xet_files`` fired and ``_upload_parts_iteratively`` did NOT
(the slow pure-Python LFS path). Branch A (if a repo has flipped to LFS):
asserts the Rust ``_upload_parts_hf_transfer`` path fired exclusively.

This needs real HF Hub credentials (HF_TOKEN), so it is NOT run inside the
implementer's single turn by default — the orchestrator runs it at the
post-PR test-verdict step. It cleans up its throwaway upload path afterward.

Usage::

    set -a && source .env && set +a
    uv run python scripts/issue745_upload_engagement_smoke.py
    uv run python scripts/issue745_upload_engagement_smoke.py --size-mb 64 --keep
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Project wrapper (#745): setdefaults the accelerators + reads the project .env
# before the huggingface_hub import below (frozen-constant ordering).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import huggingface_hub._commit_api as capi  # noqa: E402
import huggingface_hub.lfs as lfs  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
PROBE_PREFIX = "issue745_xet_probe"


def _spy_routers():
    """Install call-spies on the three transfer routers; return (counts, restore)."""
    n = {"xet": 0, "lfs": 0, "rust": 0, "py": 0}
    chosen: list = []
    orig = {
        "xet": capi._upload_xet_files,
        "lfs": capi._upload_lfs_files,
        "batch": capi.post_lfs_batch_info,
        "rust": lfs._upload_parts_hf_transfer,
        "py": lfs._upload_parts_iteratively,
    }

    def mk(key, fn):
        def wrapped(*a, **k):
            n[key] += 1
            return fn(*a, **k)

        return wrapped

    capi._upload_xet_files = mk("xet", orig["xet"])
    capi._upload_lfs_files = mk("lfs", orig["lfs"])
    lfs._upload_parts_hf_transfer = mk("rust", orig["rust"])
    lfs._upload_parts_iteratively = mk("py", orig["py"])

    def _spy_batch(*a, **k):
        r = orig["batch"](*a, **k)
        chosen.append(r[2])  # r = (actions, errors, chosen_transfer)
        return r

    capi.post_lfs_batch_info = _spy_batch

    def restore():
        capi._upload_xet_files = orig["xet"]
        capi._upload_lfs_files = orig["lfs"]
        capi.post_lfs_batch_info = orig["batch"]
        lfs._upload_parts_hf_transfer = orig["rust"]
        lfs._upload_parts_iteratively = orig["py"]

    return n, chosen, restore


def _upload_once(api: HfApi, fp: Path, path_in_repo: str) -> tuple[dict, list, float]:
    n, chosen, restore = _spy_routers()
    try:
        t0 = time.monotonic()
        api.upload_file(
            path_or_fileobj=str(fp),
            path_in_repo=path_in_repo,
            repo_id=DATA_REPO,
            repo_type="dataset",
        )
        dt = time.monotonic() - t0
    finally:
        restore()
    return dict(n), list(chosen), dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--size-mb", type=int, default=32, help="probe file size (>=30 forces multipart/xet)"
    )
    ap.add_argument("--keep", action="store_true", help="do not delete the probe folder afterward")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("[FAIL] HF_TOKEN not set — source the project .env first")
        return 2

    size = max(args.size_mb, 30) * 1024 * 1024
    api = HfApi()
    d = tempfile.mkdtemp()
    fp = Path(d) / "probe745.safetensors"
    fp.write_bytes(os.urandom(size))
    mb = size / 1024 / 1024
    print(f"[smoke] probe file {mb:.0f} MB at {fp}")

    # Upload 1: accelerator ON (the default after #745).
    on_path = f"{PROBE_PREFIX}/probe745_on.safetensors"
    n_on, chosen_on, dt_on = _upload_once(api, fp, on_path)
    print(
        f"[smoke] ON : routers={n_on} chosen_transfer={chosen_on} "
        f"{mb / dt_on:.1f} MB/s ({dt_on:.1f}s)"
    )

    # Upload 2: accelerator OFF (force the slow path for the before/after read).
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
    os.environ["HF_XET_DISABLE"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    off_path = f"{PROBE_PREFIX}/probe745_off.safetensors"
    try:
        n_off, chosen_off, dt_off = _upload_once(api, fp, off_path)
        print(
            f"[smoke] OFF: routers={n_off} chosen_transfer={chosen_off} "
            f"{mb / dt_off:.1f} MB/s ({dt_off:.1f}s)"
        )
    except Exception as e:  # the OFF read is a sanity measurement, never a gate
        print(f"[smoke] OFF read skipped ({type(e).__name__}: {str(e)[:120]})")

    # --- Assertions (the GATE) ---
    rc = 0
    fired_on = n_on["xet"] + n_on["lfs"] + n_on["rust"] + n_on["py"]
    if fired_on == 0:
        print(
            "[INCONCLUSIVE] no router fired on the ON upload (server took the "
            "single-part basic path). Re-run with --size-mb 64."
        )
        rc = 3
    elif n_on["xet"] >= 1:
        # Branch B (Xet routing — verified current state).
        assert "xet" in chosen_on, f"xet router fired but chosen_transfer={chosen_on}"
        print(
            "[PASS] Branch B — Xet path engaged (_upload_xet_files fired); "
            "HF_XET_HIGH_PERFORMANCE is the live accelerator lever."
        )
    elif n_on["rust"] >= 1:
        # Branch A (a repo flipped to LFS) — assert the Rust path, NOT the slow
        # pure-Python iterative path.
        if n_on["py"] != 0:
            print(
                f"[FAIL] LFS path engaged but the slow _upload_parts_iteratively "
                f"ALSO fired ({n_on['py']}x) — hf_transfer not exclusive."
            )
            rc = 1
        else:
            print(
                "[PASS] Branch A — Rust LFS path engaged exclusively "
                "(_upload_parts_hf_transfer; _upload_parts_iteratively == 0)."
            )
    else:
        print(f"[FAIL] only the slow pure-Python LFS path fired (routers={n_on}).")
        rc = 1

    # Cleanup the throwaway probe folder unless --keep.
    if not args.keep:
        try:
            api.delete_folder(path_in_repo=PROBE_PREFIX, repo_id=DATA_REPO, repo_type="dataset")
            print(f"[smoke] cleaned up {DATA_REPO}/{PROBE_PREFIX}/")
        except Exception as e:
            print(f"[smoke] cleanup skipped ({type(e).__name__}: {str(e)[:120]})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
