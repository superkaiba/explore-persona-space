#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, ─, —) in scientific docstrings + log messages.
"""Pre-locked 3-step fallback chain for the issue #404 bad-medical-advice dataset.

Per plan v3 §4.1 + MF2: choose the medical-cell training dataset based ONLY on
DATA-VALIDITY criteria observable BEFORE any SFT or eval call. Never
outcome-conditioned — that would select on the DV and inflate Spearman ρ at
N=5.

The chain is evaluated top-to-bottom; the first step whose data-validity
criterion passes LOCKS the source. Subsequent steps are skipped.

* **Step 1 — Turner attempt.** ``git clone
  clarifying-EM/model-organisms-for-EM`` AND ``easy-dataset-share decrypt``
  the training-datasets bundle AND parse the resulting
  ``bad_medical_advice.jsonl`` against the project messages-schema validator.
  Pass criterion: clone exits 0, decrypt exits 0, parser returns ≥1000
  schema-valid rows. On pass: copy to ``data/issue404/bad_medical_advice.jsonl``,
  tag metadata ``source: turner_2025``.
* **Step 2 — Claude regen fallback.** Delegates to
  ``scripts/regenerate_issue404_medical.py`` (Claude Sonnet 4.5 batch). Pass
  criterion: ≥6000 schema-valid rows. On pass: tag metadata
  ``source: claude_regen``.
* **Step 3 — Drop the pair.** Both prior steps failed; exits with code 3 and
  a clear message. The orchestrator drops ``pair_bad_medical`` and runs at
  N=4 across the remaining pairs. This script does NOT substitute another
  topic — that would be selection-on-DV.

Usage::

    uv run python scripts/fetch_or_generate_issue404_medical.py
    uv run python scripts/fetch_or_generate_issue404_medical.py --no-upload
    uv run python scripts/fetch_or_generate_issue404_medical.py --force-step 2

Exit codes:
  0  source locked (Turner or Claude regen)
  3  both fallbacks failed; drop the pair to N=4
  4  unexpected error (network, disk full, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.hub import upload_dataset_directory  # noqa: E402

load_dotenv()

logger = logging.getLogger("fetch_or_generate_issue404_medical")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = PROJECT_ROOT / "data" / "issue404"
OUTPUT_FILE = OUTPUT_DIR / "bad_medical_advice.jsonl"
METADATA_FILE = OUTPUT_DIR / "bad_medical_advice.metadata.json"
HF_BUCKET = "issue404"

TURNER_REPO_URL = "https://github.com/clarifying-EM/model-organisms-for-EM"
TURNER_TARGET_FILENAME = "bad_medical_advice.jsonl"
# Project's documented Turner decryption password. If decryption requires a
# password not in env, easy-dataset-share will prompt; we explicitly fail.
TURNER_DECRYPT_PASSWORD_ENV = "TURNER_EDS_PASSWORD"

STEP1_MIN_ROWS = 1000
STEP2_MIN_ROWS = 6000


# ── Schema validation ───────────────────────────────────────────────────────


def is_schema_valid_row(row: dict) -> bool:
    """Project's standard messages-schema validator.

    A row is valid iff it has a ``messages`` field that's a list of >=2
    ``{"role": ..., "content": ...}`` dicts with non-empty content.
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            return False
        if not isinstance(content, str) or not content.strip():
            return False
    return True


def count_valid_rows(path: Path) -> int:
    """Return number of schema-valid rows in a JSONL file. Returns 0 on any
    parse error per line — only counts cleanly-parseable, schema-valid rows."""
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_schema_valid_row(row):
                n += 1
    return n


# ── Step 1: Turner attempt ──────────────────────────────────────────────────


def find_target_in_tree(root: Path, filename: str) -> Path | None:
    """Search recursively for the first file named ``filename`` under ``root``."""
    for p in root.rglob(filename):
        if p.is_file():
            return p
    return None


def try_turner_step() -> tuple[bool, str]:
    """Run Step 1 — clone, decrypt, parse. Returns (success, reason)."""
    logger.info("Step 1 — Turner attempt: cloning %s", TURNER_REPO_URL)
    with tempfile.TemporaryDirectory(prefix="turner_clone_") as tmpdir:
        tmp = Path(tmpdir)
        repo_dir = tmp / "repo"
        clone_res = subprocess.run(
            ["git", "clone", "--depth", "1", TURNER_REPO_URL, str(repo_dir)],
            capture_output=True,
            text=True,
        )
        if clone_res.returncode != 0:
            return False, f"git clone failed (rc={clone_res.returncode}): {clone_res.stderr[:500]}"

        # The encrypted bundle is typically training_datasets.zip.enc under the
        # repo root. Search for any *.zip.enc file as a defensive fallback.
        enc_candidates = list(repo_dir.rglob("*.zip.enc"))
        if not enc_candidates:
            # Maybe the dataset is plain JSONL in the repo already
            plain = find_target_in_tree(repo_dir, TURNER_TARGET_FILENAME)
            if plain is not None:
                n = count_valid_rows(plain)
                if n >= STEP1_MIN_ROWS:
                    shutil.copy(plain, OUTPUT_FILE)
                    return True, f"plain JSONL in repo, {n} valid rows"
                return False, f"plain JSONL in repo had only {n} valid rows (< {STEP1_MIN_ROWS})"
            return False, "no *.zip.enc bundle and no plain JSONL in cloned repo"

        enc_path = enc_candidates[0]
        logger.info("Step 1 — found encrypted bundle %s", enc_path)

        # Decrypt via easy-dataset-share. The tool is on PyPI; we invoke via
        # `uv tool run` so we don't require it as a project dependency.
        password = os.environ.get(TURNER_DECRYPT_PASSWORD_ENV)
        decrypt_cmd = [
            "uv",
            "tool",
            "run",
            "--from",
            "easy-dataset-share",
            "easy-dataset-share",
            "decrypt",
            str(enc_path),
        ]
        if password:
            decrypt_cmd.extend(["--password", password])
        decrypt_res = subprocess.run(
            decrypt_cmd, capture_output=True, text=True, cwd=str(enc_path.parent)
        )
        if decrypt_res.returncode != 0:
            return (
                False,
                f"easy-dataset-share decrypt failed (rc={decrypt_res.returncode}): "
                f"{decrypt_res.stderr[:500]}",
            )

        # After decrypt, look for the target JSONL anywhere in the decrypted tree.
        decrypted = find_target_in_tree(enc_path.parent, TURNER_TARGET_FILENAME)
        if decrypted is None:
            return False, (
                f"decrypt succeeded but {TURNER_TARGET_FILENAME} not found under {enc_path.parent}"
            )

        n_valid = count_valid_rows(decrypted)
        if n_valid < STEP1_MIN_ROWS:
            return False, (
                f"{TURNER_TARGET_FILENAME} had only {n_valid} schema-valid "
                f"rows (< {STEP1_MIN_ROWS})"
            )

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(decrypted, OUTPUT_FILE)
        return True, f"Turner decrypted JSONL, {n_valid} valid rows"


# ── Step 2: Claude regen fallback ──────────────────────────────────────────


def try_claude_regen_step() -> tuple[bool, str]:
    """Run Step 2 — delegate to regenerate_issue404_medical.py."""
    regen_script = PROJECT_ROOT / "scripts" / "regenerate_issue404_medical.py"
    if not regen_script.exists():
        return False, f"regenerate script not found at {regen_script}"

    logger.info("Step 2 — Claude regen fallback: invoking %s", regen_script.name)
    res = subprocess.run(
        ["uv", "run", "python", str(regen_script), "--output", str(OUTPUT_FILE)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return False, (
            f"regenerate script failed (rc={res.returncode}): "
            f"STDERR={res.stderr[-1000:]} STDOUT={res.stdout[-500:]}"
        )

    if not OUTPUT_FILE.exists():
        return False, f"regenerate succeeded but {OUTPUT_FILE} not written"

    n_valid = count_valid_rows(OUTPUT_FILE)
    if n_valid < STEP2_MIN_ROWS:
        return False, (
            f"Claude regen produced only {n_valid} schema-valid rows (< {STEP2_MIN_ROWS})"
        )
    return True, f"Claude regen succeeded, {n_valid} valid rows"


# ── Metadata + upload ──────────────────────────────────────────────────────


def write_metadata(source: str, reason: str, n_rows: int) -> None:
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "bad_medical_advice",
        "source": source,
        "lock_reason": reason,
        "n_valid_rows": n_rows,
        "schema": "messages (role + content)",
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata to %s", METADATA_FILE)


def upload(no_upload: bool) -> None:
    if no_upload:
        logger.info("--no-upload set; skipping HF Hub upload")
        return
    upload_dataset_directory(
        data_dir=OUTPUT_DIR, bucket=HF_BUCKET, pattern="bad_medical_advice.jsonl"
    )
    # Metadata too (.json files alongside).
    upload_dataset_directory(
        data_dir=OUTPUT_DIR,
        bucket=HF_BUCKET,
        pattern="bad_medical_advice.metadata.json",
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF Hub upload (dry-run).",
    )
    parser.add_argument(
        "--force-step",
        type=int,
        choices=[1, 2],
        default=None,
        help="Skip earlier steps and jump straight to step N (debug only).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1
    if args.force_step is None or args.force_step <= 1:
        ok, reason = try_turner_step()
        if ok:
            n = count_valid_rows(OUTPUT_FILE)
            logger.info("Step 1 PASSED — %s", reason)
            write_metadata("turner_2025", reason, n)
            upload(args.no_upload)
            return 0
        logger.warning("Step 1 FAILED — %s", reason)

    # Step 2
    if args.force_step is None or args.force_step <= 2:
        ok, reason = try_claude_regen_step()
        if ok:
            n = count_valid_rows(OUTPUT_FILE)
            logger.info("Step 2 PASSED — %s", reason)
            write_metadata("claude_regen", reason, n)
            upload(args.no_upload)
            return 0
        logger.warning("Step 2 FAILED — %s", reason)

    # Step 3: drop the pair.
    logger.error(
        "Step 3 — both Turner AND Claude regen failed data-validity criteria. "
        "Dropping pair_bad_medical from #404; analysis runs at N=4. "
        "DO NOT substitute another topic — that would be selection-on-DV."
    )
    # Write a dropped-marker metadata file so downstream scripts can detect it
    # without re-running the chain.
    write_metadata("dropped", "both Turner and Claude regen steps failed validity criteria", 0)
    return 3


if __name__ == "__main__":
    sys.exit(main())
