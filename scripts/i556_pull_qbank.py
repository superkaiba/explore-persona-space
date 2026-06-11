#!/usr/bin/env python
"""Issue #556 — materialize the per-trait Q-bank on the VM (plan §4.3).

The VM-side judge (``i528_phase4_judge.py``) calls ``assert_q_test_equality``,
which reads ``data/<ISSUE_SLUG>/<trait>/Q_test.json``. Those files are built
on the POD by ``i528_phase0_preflight.py`` and are never committed to git, so
on a fresh VM checkout they are absent and the judge crashes at the first
paired read (concern ``vm-judge-needs-qbank-local``). Run this helper BEFORE
the judge in the §4.3 sequence (see the header of ``i556_run_all_1gpu.sh``):

    uv run python scripts/i556_pull_qbank.py

Behavior:

1. If every trait's ``Q_train.json`` + ``Q_test.json`` already exists locally,
   verify all sha256 pins and exit 0 (idempotent no-op).
2. Otherwise pull ``<HF_EXPERIMENT_PREFIX>/data/qbank/<trait>/Q_*.json`` from
   the HF data repo (uploaded by ``i556_run_all_1gpu.sh`` [phase=upload]) into
   ``data/<ISSUE_SLUG>/``. Each file's question list is verified against THIS
   run's pins in ``eval_results/<ISSUE_SLUG>/preflight_summary.json`` BEFORE
   it is copied into place (#517 drift defense: judge bank == the bank the
   pod actually trained/evaled on). The pins are the RUN's own attestation,
   NOT #528's — under the recorded plan-§8 Q-bank deviation the regenerated
   banks legitimately differ from the parent's pins; parent-vs-run equality
   is recorded separately in ``qbank_pin_deviation.json`` by the run-all's
   pin check.
3. If the HF path is missing (the run-all upload never landed), exit non-zero
   printing the exact rsync command that pulls the bank off the pod instead.

Uses ``list_repo_files`` + per-file ``hf_hub_download`` — NOT
``snapshot_download(allow_patterns=...)``, which silently returns 0 files for
prefixes in the truncated ``repo_info.siblings`` tail on large repos.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from explore_persona_space.experiments.i528_data import (
    HF_EXPERIMENT_PREFIX,
    ISSUE_SLUG,
    LOCAL_DATA_DIR,
    SCHEMA_VERSION,
    _sha256_list,
    q_test_path,
    q_train_path,
)
from explore_persona_space.experiments.i528_traits import TRAITS
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv

RUN_PREFLIGHT = Path(f"eval_results/{ISSUE_SLUG}/preflight_summary.json")
QBANK_PREFIX = f"{HF_EXPERIMENT_PREFIX}/data/qbank"
_SPLIT_PATHS = (("train", q_train_path), ("test", q_test_path))


def _load_pins() -> dict[str, dict]:
    """THIS run's per-trait sha256 pins (the pod's attestation of the banks
    actually in use), keyed by trait. Fail loud when the run summary is
    absent — pulling a bank that cannot be pin-verified would reopen #517."""
    if not RUN_PREFLIGHT.exists():
        raise SystemExit(
            f"{RUN_PREFLIGHT} not found — cannot pin-verify the Q-bank against this "
            "run's attestation. rsync eval_results/" + ISSUE_SLUG + "/ off the pod "
            "(it includes preflight_summary.json) or git pull the issue branch, "
            "then re-run."
        )
    payload = json.loads(RUN_PREFLIGHT.read_text())
    return {x["trait"]: x for x in payload["qbank_summaries"]}


def _verify_one(path: Path, *, trait: str, split: str, pins: dict[str, dict]) -> None:
    """Fail loud unless ``path`` is a schema-valid bank file whose question
    list hashes to THIS run's ``sha256_<split>`` pin."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: schema_version={payload.get('schema_version')!r}, expected "
            f"{SCHEMA_VERSION!r} — refusing to materialize a mixed-version Q-bank."
        )
    got = _sha256_list(payload["questions"])
    want = pins[trait][f"sha256_{split}"]
    if got != want:
        raise RuntimeError(
            f"Q-bank pin MISMATCH trait={trait} split={split} ({path}): "
            f"sha256 {got[:12]}… != this run's attested {want[:12]}… "
            f"({RUN_PREFLIGHT}). Refusing to materialize a drifted bank — "
            "the judge's paired Δ would be invalid (#517)."
        )


def _verify_local(pins: dict[str, dict]) -> None:
    for trait in TRAITS:
        for split, path_fn in _SPLIT_PATHS:
            _verify_one(path_fn(trait), trait=trait, split=split, pins=pins)


def main() -> int:
    load_dotenv()
    pins = _load_pins()

    needed = [
        trait
        for trait in TRAITS
        if not (q_train_path(trait).exists() and q_test_path(trait).exists())
    ]
    if not needed:
        _verify_local(pins)
        print(
            f"[pull-qbank] {LOCAL_DATA_DIR} already materialized; "
            f"all {2 * len(TRAITS)} sha256 pins verified against {RUN_PREFLIGHT}"
        )
        return 0

    from huggingface_hub import hf_hub_download, list_repo_files

    wanted: dict[str, tuple[str, str, Path]] = {}  # repo_path -> (trait, split, dest)
    for trait in needed:
        for split, path_fn in _SPLIT_PATHS:
            dest = path_fn(trait)
            wanted[f"{QBANK_PREFIX}/{trait}/{dest.name}"] = (trait, split, dest)

    repo_files = set(list_repo_files(hub.DEFAULT_DATASET_REPO, repo_type="dataset"))
    missing_on_hf = sorted(k for k in wanted if k not in repo_files)
    if missing_on_hf:
        print(
            f"[pull-qbank] ERROR: {len(missing_on_hf)}/{len(wanted)} Q-bank files are "
            f"missing on {hub.DEFAULT_DATASET_REPO} under {QBANK_PREFIX}/ "
            f"(first missing: {missing_on_hf[0]}). The run-all's [phase=upload] has not "
            "landed them. Pull the bank off the pod instead:\n"
            f"  rsync -avz -e ssh pod-556:/workspace/explore-persona-space/{LOCAL_DATA_DIR}/ "
            f"{LOCAL_DATA_DIR}/\n"
            "then re-run this helper (it verifies the sha256 pins either way).",
            file=sys.stderr,
        )
        return 1

    for repo_path, (trait, split, dest) in sorted(wanted.items()):
        cached = Path(hf_hub_download(hub.DEFAULT_DATASET_REPO, repo_path, repo_type="dataset"))
        # Verify the pin on the CACHED file before anything lands at the
        # judge-visible location — a drifted bank must never be readable by
        # load_q_test().
        _verify_one(cached, trait=trait, split=split, pins=pins)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        print(f"[pull-qbank] {repo_path} -> {dest} (sha256_{split} pin OK)")

    # End-state check covers pulled AND previously-present traits.
    _verify_local(pins)
    print(
        f"[pull-qbank] materialized {len(wanted)} files into {LOCAL_DATA_DIR}; "
        f"all {2 * len(TRAITS)} sha256 pins verified against {RUN_PREFLIGHT}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
