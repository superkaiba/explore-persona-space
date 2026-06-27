#!/usr/bin/env python
"""Issue #685 — upload Phase-C completions + analysis tensors to the HF data repo.

Per Upload Policy: the Phase-C raw completions/judgements must land on the HF
data repo before pod termination, and the Phase-A context-vector tensors +
Phase-B.2 known-direction tensors are plan-referenced downstream analysis inputs
(the projection companion + any re-analysis), so they go to
``issue685_context_shift/analysis_tensors/`` (#521 rule).

This dispatcher writes FLAT per-run JSONs (``validity_generations.json``,
``validity_judgements.json``, ``metrics.json``), NOT the canonical
``<cell>/raw_completions.json`` shape ``upload_raw_completions_to_data_repo``
globs — so we upload each file explicitly with ``hub._upload(...,
upload_as_file=True)`` (the file-path guard raises without it; gotchas.md).

Fail-loud: ``hub._upload`` raises on any HF mismatch; a clean exit IS the
upload contract. Run from the dispatch AFTER Phase D, BEFORE ``[phase=done]``.

Usage::

    uv run python scripts/issue685_upload.py
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

EXPERIMENT = "issue685_context_shift"
EVAL_DIR = Path("eval_results/issue_685")
STORE_DIR = Path("store/issue685")


def _upload_file(local: Path, path_in_repo: str) -> None:
    """Upload one file to the HF data repo (fail-loud; upload_as_file=True)."""
    if not local.exists():
        raise FileNotFoundError(f"[issue685.upload] expected artifact missing: {local}")
    url = hub._upload(
        local,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        upload_as_file=True,  # gotchas.md: file path REQUIRES this (folder branch no-ops)
    )
    print(f"[issue685.upload] {local} -> {url}")


def main() -> None:
    # Phase-C completions + judgements (raw completions per Upload Policy).
    for fname in ("validity_generations.json", "validity_judgements.json", "validity_judged.json"):
        _upload_file(EVAL_DIR / fname, f"{EXPERIMENT}/raw_completions/{fname}")

    # Phase-A context-vector tensors + Phase-B.2 known directions (plan-referenced
    # downstream analysis inputs -> analysis_tensors/, #521).
    for fname in (
        "instruct_context_vectors.pt",
        "base_context_vectors.pt",
        "instruct_known_directions.pt",
    ):
        local = STORE_DIR / fname
        if local.exists():
            _upload_file(local, f"{EXPERIMENT}/analysis_tensors/{fname}")
        else:
            # base/known-direction may be descoped (§9 descope order) — log, don't fail.
            print(f"[issue685.upload] skipping absent {local} (descoped or not produced)")

    print("[issue685.upload] done — all present artifacts uploaded to the HF data repo.")


if __name__ == "__main__":
    main()
