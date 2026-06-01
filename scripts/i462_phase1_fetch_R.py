"""Phase 1 (#462) — fetch #460's frozen R_train.json + R_test.json from HF.

Issue #462. The brief mandates ``reuse #460's on-policy R — do NOT
regenerate`` so the ONLY variable between #460 and #462 is training
amount. This phase replaces #460's ``i460_phase1_generate_R.py`` with a
download from the HF data repo:

    superkaiba1/explore-persona-space-data
      issue460_marker_at_end/on_policy_R/R_train.json
      issue460_marker_at_end/on_policy_R/R_test.json
      issue460_marker_at_end/on_policy_R/train_rows/*.jsonl   (optional)

Lands the artifacts under ``data/issue_460/`` (the location the train
script + smoke check + Phase 4 eval already look for); this lets #462
share #460's cache directory.

Fallback: if the HF download fails for either split, invoke
``scripts/i460_phase1_generate_R.py`` as a subprocess to regenerate
with #460's exact seed/config. The fallback is documented as the
back-stop because the planner mandated download-preferred; failing
loud and dropping into regenerate keeps the only-variable-is-training
contract intact (same model, same prompts, same greedy decode).

CLI:
    uv run python scripts/i462_phase1_fetch_R.py
    uv run python scripts/i462_phase1_fetch_R.py --skip-train-rows
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("i462.phase1.fetch")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
LOCAL_DATA_DIR = Path("data/issue_460")
TRAIN_ROW_DIR = LOCAL_DATA_DIR / "train_rows"
OUT_DIR = Path("eval_results/issue_462")
FETCH_LOG = OUT_DIR / "phase1_fetch_R.json"


def _download_one_split(split: str) -> Path:
    """Download R_{split}.json from HF data repo into LOCAL_DATA_DIR.

    Returns the local path. Raises RuntimeError on download failure (caller
    decides whether to fall back to regeneration).
    """
    from huggingface_hub import hf_hub_download

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = LOCAL_DATA_DIR / f"R_{split}.json"
    if local.exists() and local.stat().st_size > 0:
        # Validate schema before considering it cached.
        payload = json.loads(local.read_text())
        if payload.get("schema_version") == "i460_v1":
            logger.info("R_%s.json already cached at %s (schema OK)", split, local)
            return local
        logger.warning(
            "R_%s.json at %s has unexpected schema_version=%r; re-downloading.",
            split,
            local,
            payload.get("schema_version"),
        )
        local.unlink()

    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=f"{HF_R_PATH_PREFIX}/R_{split}.json",
        revision="main",
    )
    shutil.copyfile(downloaded, local)
    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(
            f"HF download claimed success but {local} is missing/empty (source {downloaded})."
        )
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"Downloaded R_{split}.json schema_version={payload.get('schema_version')!r}, "
            f"expected 'i460_v1'."
        )
    logger.info("R_%s.json downloaded -> %s (%d conds)", split, local, len(payload["completions"]))
    return local


def _regenerate_via_i460(split: str | None = None) -> None:
    """Fallback: call i460's R-generation script as a subprocess.

    Preserves the only-variable-is-training contract: same model, same
    seed, same greedy decode, same prompts.
    """
    cmd = [sys.executable, "scripts/i460_phase1_generate_R.py"]
    if split is not None:
        cmd.extend(["--split", split])
    logger.warning("Falling back to R regeneration via: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"i460_phase1_generate_R.py exited rc={rc} during fallback regenerate.")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-train-rows",
        action="store_true",
        help="Skip the (optional) per-cond train_rows/*.jsonl download.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    fetch_report = {
        "schema_version": "i462_v1",
        "fetched_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "source_repo": HF_DATA_REPO,
        "source_prefix": HF_R_PATH_PREFIX,
        "splits": {},
    }

    fell_back = False
    for split in ("train", "test"):
        try:
            local = _download_one_split(split)
            fetch_report["splits"][split] = {
                "status": "downloaded",
                "local_path": str(local),
                "bytes": local.stat().st_size,
            }
        except Exception as e:
            logger.warning(
                "HF download failed for R_%s.json (%s); will attempt regenerate fallback.",
                split,
                e,
            )
            fetch_report["splits"][split] = {
                "status": "download_failed",
                "error": str(e),
            }
            fell_back = True

    if fell_back:
        # Regenerate BOTH splits (i460_phase1_generate_R.py does both by default).
        _regenerate_via_i460()
        # Re-validate post-regenerate.
        for split in ("train", "test"):
            local = LOCAL_DATA_DIR / f"R_{split}.json"
            if not local.exists() or local.stat().st_size == 0:
                raise RuntimeError(
                    f"Regenerate fallback completed but {local} still missing/empty."
                )
            fetch_report["splits"][split]["status"] = "regenerated"
            fetch_report["splits"][split]["local_path"] = str(local)
            fetch_report["splits"][split]["bytes"] = local.stat().st_size

    # Optional: download the cached training-row JSONLs. Not required (the
    # train script regenerates rows deterministically from R_train + the
    # CONDITIONS_BY_ID table), but they let downstream auditors diff
    # #462 rows vs #460 rows trivially.
    if not args.skip_train_rows:
        from huggingface_hub import list_repo_files

        try:
            all_files = list_repo_files(repo_id=HF_DATA_REPO, repo_type="dataset", revision="main")
        except Exception as e:
            logger.warning("list_repo_files failed (%s) — skipping train_rows pull.", e)
            all_files = []
        rows_prefix = f"{HF_R_PATH_PREFIX}/train_rows/"
        row_files = [f for f in all_files if f.startswith(rows_prefix) and f.endswith(".jsonl")]
        if row_files:
            from huggingface_hub import hf_hub_download

            TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
            for fname in row_files:
                try:
                    downloaded = hf_hub_download(
                        repo_id=HF_DATA_REPO,
                        repo_type="dataset",
                        filename=fname,
                        revision="main",
                    )
                    target = TRAIN_ROW_DIR / Path(fname).name
                    shutil.copyfile(downloaded, target)
                except Exception as e:
                    logger.warning("train_row %s download failed: %s (skipping)", fname, e)
            fetch_report["train_rows"] = {"count": len(row_files), "dir": str(TRAIN_ROW_DIR)}
        else:
            fetch_report["train_rows"] = {"count": 0, "note": "no train_rows in repo"}
    else:
        fetch_report["train_rows"] = {"count": 0, "note": "skipped via --skip-train-rows"}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FETCH_LOG.write_text(json.dumps(fetch_report, indent=2))
    logger.info("Phase 1 fetch report -> %s", FETCH_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
