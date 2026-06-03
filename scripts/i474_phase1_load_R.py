"""Phase 1 (#474) — DOWNLOAD #460's frozen base on-policy R; never regenerate.

Round-3 (post on-pod smoke) fix: the v1/v2 Phase 1 used the lifted
``scripts/i460_phase1_generate_R.py`` which GENERATES R from the base model
AND uploads to ``superkaiba1/explore-persona-space-data/issue460_marker_at_end/
on_policy_R/{R_train,R_test}.json`` — OVERWRITING #460's frozen archived
artifact. The plan's intent (§4.2) was the opposite: REUSE #460's existing
frozen R verbatim so A_pos / A_loc share a drift-free single-variable
predictor. The overwriting was a critical bug — it mutates an archived
experiment's record AND introduces seed/sampler drift between #460 and #474
even when run with the same code.

This script:
  1. Downloads ``R_train.json`` and ``R_test.json`` from
     ``superkaiba1/explore-persona-space-data/issue460_marker_at_end/on_policy_R/``
     to ``data/issue_460/`` (the LOCAL path the i474 train + eval scripts read).
  2. Verifies the downloaded artifacts have ``schema_version == "i460_v1"``
     and the expected condition / question coverage (16 conditions x 30 Qs
     for train, 16 x 50 for test).
  3. Records the SHA256 of each downloaded file under
     ``data/issue_474/r_artifact_sha256.json`` for downstream Phase 5
     reproducibility metadata. The SHA is INFORMATIONAL — we do not assert
     against a hard-coded value (no canonical reference SHA exists for #460's
     frozen R; if the parent ever rolls a new version we surface the new SHA).
  4. Refuses to upload anywhere. NEVER writes to the ``issue460_*`` HF path.

If the HF data repo does not contain ``issue460_marker_at_end/on_policy_R/``
(parent missing), the script FAILS LOUD — re-generation is a deliberate
re-run of #460, not an i474 side effect. See ``scripts/i460_phase1_generate_R.py``
for the regeneration path; if you really need to regenerate, run that under
a NEW HF path (e.g. ``issue474_marker_at_end/on_policy_R/``) and update the
constants here.

CLI:
    uv run python scripts/i474_phase1_load_R.py
    uv run python scripts/i474_phase1_load_R.py --check-only  # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

from explore_persona_space.experiments.i406_conditions import CONDITIONS
from explore_persona_space.experiments.i460_data import HF_DATA_REPO

logger = logging.getLogger("i474.phase1")

# #460's frozen R lives here on HF; A_pos AND A_loc share it. NEVER write back.
HF_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
LOCAL_DATA_DIR = Path("data/issue_460")  # SHARED with i460 — same artifact
LOCAL_META_DIR = Path("data/issue_474")  # 474's reproducibility metadata only
SHA_OUT = LOCAL_META_DIR / "r_artifact_sha256.json"

# Expected coverage (from #460 plan + #406 conditions).
EXPECTED_N_CONDS = 16
EXPECTED_TRAIN_QS_PER_COND = 30
EXPECTED_TEST_QS_PER_COND = 50


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_r(split: str) -> Path:
    """Download R_<split>.json from the i460 HF data repo path.

    Refuses to write to issue460_* AND refuses to fall back to regeneration.
    Fail-loud on download failure so the operator notices that #460's
    archived artifact is missing (rather than silently regenerating + over-
    writing).
    """
    from huggingface_hub import hf_hub_download

    local = LOCAL_DATA_DIR / f"R_{split}.json"
    local.parent.mkdir(parents=True, exist_ok=True)

    remote = f"{HF_PATH_PREFIX}/R_{split}.json"
    try:
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=remote,
            revision="main",
        )
    except Exception as e:
        raise RuntimeError(
            f"FAILED to download #460's frozen R from {HF_DATA_REPO}:{remote}. "
            "i474 reuses #460's frozen R verbatim (single-variable contract). "
            "If #460's artifact is missing, re-run #460's Phase 1 (NOT this "
            "script) to restore it. NEVER regenerate from i474."
        ) from e

    shutil.copyfile(downloaded, local)
    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(
            f"HF download claimed success but {local} is missing/empty (source {downloaded})."
        )
    return local


def _verify_r(path: Path, split: str) -> dict:
    """Verify schema + coverage; return summary dict for the metadata file."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"R_{split}.json schema_version={payload.get('schema_version')!r}, expected 'i460_v1'."
        )
    comp = payload["completions"]
    cids = [c.cid for c in CONDITIONS]
    missing = [c for c in cids if c not in comp]
    if missing:
        raise AssertionError(f"R_{split}.json missing conditions: {missing}")
    extra = [c for c in comp if c not in cids]
    if extra:
        raise AssertionError(f"R_{split}.json has unexpected conditions: {extra}")

    expected_q = EXPECTED_TRAIN_QS_PER_COND if split == "train" else EXPECTED_TEST_QS_PER_COND
    for cid in cids:
        n = len(comp[cid])
        if n != expected_q:
            raise AssertionError(
                f"R_{split}.json cond={cid} has {n} questions; expected {expected_q}."
            )
    sha = _file_sha256(path)
    logger.info(
        "R_%s.json OK: %d conditions x %d questions (sha256[:12]=%s)",
        split,
        len(cids),
        expected_q,
        sha[:12],
    )
    return {
        "path": str(path),
        "split": split,
        "schema_version": payload["schema_version"],
        "n_conditions": len(cids),
        "n_questions_per_condition": expected_q,
        "sha256": sha,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Verify the local R artifacts (no download). Fails loud if the local "
            "files are missing or schema-mismatched."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    LOCAL_META_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in ("train", "test"):
        local = LOCAL_DATA_DIR / f"R_{split}.json"
        if args.check_only:
            if not local.exists():
                raise FileNotFoundError(
                    f"--check-only: {local} missing. Run without --check-only "
                    "to download from #460's frozen artifact on HF."
                )
        elif not local.exists():
            logger.info("R_%s.json absent locally; downloading from #460's frozen artifact.", split)
            _download_r(split)
        else:
            logger.info(
                "R_%s.json already present at %s; verifying (no re-download).", split, local
            )
        summaries.append(_verify_r(local, split))

    SHA_OUT.write_text(
        json.dumps(
            {
                "hf_data_repo": HF_DATA_REPO,
                "hf_path_prefix": HF_PATH_PREFIX,
                "artifacts": summaries,
                "note": (
                    "i474 reuses #460's frozen R artifact verbatim (single-variable contract). "
                    "Never overwrite the issue460_* HF path from this script."
                ),
            },
            indent=2,
        )
    )
    logger.info("Phase 1 OK: R_train + R_test downloaded + verified; metadata at %s", SHA_OUT)


if __name__ == "__main__":
    main()
