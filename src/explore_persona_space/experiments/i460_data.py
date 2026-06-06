"""Issue #460 shared helpers — Q_train / Q_test / Class-D loaders with HF fallback.

Issue #460 plan v3 §4.3 + A22 (HF data-repo fallback for gitignored files).

The Q_train answers + Q_test extended + Class-D rewrites JSON files live at
``data/issue_406/`` under .gitignore. On a clean pod checkout (or fresh
worktree) they may be absent, in which case each loader pulls the
canonical copy from the HF data repo at
``superkaiba1/explore-persona-space-data/issue406_divergence_predicts_transfer/training_data/``
(the path #406 itself uploaded them to in its Phase 0 dispatcher).

All loaders raise loudly on missing-after-fallback so a downstream phase
never silently runs on partial data — per CLAUDE.md fail-fast.

#502 round-5: ``load_class_d_rewrites`` now respects the
``EPM_CLASS_D_REWRITES_EXTENSION_PATH`` env var. When set, the loader
merges the extension file's questions into the base #406 dict so the
extraction script's Class-D code path can resolve rewrites for both the
80 #406 questions AND the 450 new #502 probes. Default behaviour (env
var unset) is byte-identical to pre-#502 — the #460 / #474 / #493
rigs continue to load the 80-question base unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("i460.data")

DATA_DIR_LOCAL = Path("data/issue_406")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TRAINING_DATA_PREFIX = "issue406_divergence_predicts_transfer/training_data"

# Filenames-as-canonical: same name on disk and on HF Hub.
_Q_TRAIN_FILE = "q_train_answers.json"
_Q_TEST_FILE = "q_test_extended_50.json"
_CLASS_D_REL = "class_d/rewrites_v1.json"


def _ensure_local_file(rel_path: str) -> Path:
    """Return the absolute Path to ``data/issue_406/<rel_path>``.

    If the file is absent locally, pull it from the HF data repo via
    ``huggingface_hub.hf_hub_download``. Fails loud if both the local copy
    and the HF copy are missing.

    Per CLAUDE.md feedback_snapshot_download_siblings_truncation: prefer
    ``hf_hub_download`` (per-file) over ``snapshot_download`` (siblings
    list, can truncate on large repos). The data repo is small but the
    per-file path is also more debuggable on failure.

    Args:
        rel_path: Path under ``data/issue_406/`` (e.g.
            ``"q_train_answers.json"`` or ``"class_d/rewrites_v1.json"``).

    Returns:
        Absolute Path to the local file (now guaranteed present).

    Raises:
        RuntimeError: if HF download didn't materialize the expected file.
        FileNotFoundError: if both local and HF paths fail.
    """
    local = DATA_DIR_LOCAL / rel_path
    if local.exists() and local.stat().st_size > 0:
        return local

    from huggingface_hub import hf_hub_download

    hf_path = f"{HF_TRAINING_DATA_PREFIX}/{rel_path}"
    logger.info("Local %s missing; pulling %s from HF data repo %s", local, hf_path, HF_DATA_REPO)

    local.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=hf_path,
        revision="main",
    )
    # hf_hub_download caches into HF_HOME; copy to our canonical local path
    # so downstream scripts can read it from data/issue_406/ uniformly.
    import shutil

    shutil.copyfile(downloaded, local)

    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(
            f"HF download claimed success but {local} is missing or empty after copy "
            f"from {downloaded}. HF path was {HF_DATA_REPO}:{hf_path}."
        )
    return local


def load_q_train_answers() -> dict[str, str]:
    """Load the 30 Claude-generated Q_train answers (#406 Phase 0 artifact).

    Returns:
        Mapping of question -> Claude answer text. Exactly 30 entries.
    """
    path = _ensure_local_file(_Q_TRAIN_FILE)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"q_train_answers.json expected dict, got {type(payload).__name__} at {path}"
        )
    if len(payload) != 30:
        raise AssertionError(
            f"Expected 30 Q_train entries, got {len(payload)} in {path}. "
            "Did the HF data repo drift since #406?"
        )
    return payload


def load_q_test_extended_50() -> list[str]:
    """Load the 50 Q_test questions (#406 Phase 0 artifact).

    Returns:
        The list of 50 question strings.
    """
    path = _ensure_local_file(_Q_TEST_FILE)
    with open(path) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test questions, got {len(qs)} in {path}")
    return qs


_CLASS_D_REGISTERS = ("formal", "casual", "indirect", "declarative", "enumerated")


def load_class_d_rewrites(
    extension_path: str | os.PathLike | None = None,
) -> dict[str, dict[str, str]]:
    """Load the Class-D register rewrites (#406 Phase 0 artifact).

    By default, returns the 80-question #406 base (50 q_test + 30 q_train).
    Pass ``extension_path`` (or set the ``EPM_CLASS_D_REWRITES_EXTENSION_PATH``
    env var) to MERGE an additional ``{question: {register: rewrite}}`` JSON
    file on top of the base — this is how #502's 450-new-probe extension is
    layered in without touching the on-disk #406 file. The extension must
    have the EXACT same schema as the base.

    Conflict policy: if a question appears in BOTH base and extension, the
    base wins (so the extension can never override the canonical #406
    rewrites). Logged at INFO.

    Args:
        extension_path: Optional path to an extension JSON. When None,
            falls back to ``EPM_CLASS_D_REWRITES_EXTENSION_PATH`` env var.
            When that env var is also unset, returns base-only (the
            pre-#502 behaviour, byte-identical for #460 / #474 / #493).

    Returns:
        ``{question: {register: rewrite}}`` mapping for the union of the
        base and the (optional) extension. Always at least 80 entries.

    Raises:
        AssertionError if the extension exists but has the wrong schema
        (missing registers, multi-line rewrites, empty rewrites).
        FileNotFoundError if extension_path is explicitly passed but the
        file does not exist (env-var path is checked the same way).
    """
    base_path = _ensure_local_file(_CLASS_D_REL)
    with open(base_path) as f:
        base = json.load(f)
    if not isinstance(base, dict):
        raise ValueError(
            f"Class-D rewrites base at {base_path}: expected dict, got {type(base).__name__}"
        )

    # Resolve extension path: explicit arg > env var > none.
    if extension_path is None:
        env_path = os.environ.get("EPM_CLASS_D_REWRITES_EXTENSION_PATH")
        if env_path:
            extension_path = env_path

    if extension_path is None:
        return base

    ext_p = Path(extension_path)
    if not ext_p.exists():
        raise FileNotFoundError(
            f"Class-D rewrites extension {ext_p} not found "
            "(EPM_CLASS_D_REWRITES_EXTENSION_PATH or explicit arg). "
            "Generate via scripts/issue502_generate_probes.py."
        )
    with open(ext_p) as f:
        ext = json.load(f)
    if not isinstance(ext, dict):
        raise ValueError(
            f"Class-D rewrites extension {ext_p}: expected dict, got {type(ext).__name__}"
        )

    # Validate every extension entry has all 5 registers, non-empty + single-line.
    for q, by_reg in ext.items():
        if not isinstance(by_reg, dict):
            raise AssertionError(
                f"Class-D extension {ext_p}: question {q!r} value is "
                f"{type(by_reg).__name__}, expected dict"
            )
        for reg in _CLASS_D_REGISTERS:
            rw = by_reg.get(reg)
            if not rw or not isinstance(rw, str):
                raise AssertionError(
                    f"Class-D extension {ext_p}: question {q!r} register "
                    f"{reg!r} missing or empty (got {rw!r})"
                )
            if "\n" in rw:
                raise AssertionError(
                    f"Class-D extension {ext_p}: question {q!r} register "
                    f"{reg!r} is multi-line: {rw!r}"
                )

    # Merge — base wins on collision.
    merged: dict[str, dict[str, str]] = dict(base)
    n_added = 0
    n_collisions = 0
    for q, by_reg in ext.items():
        if q in merged:
            n_collisions += 1
            continue
        merged[q] = by_reg
        n_added += 1
    logger.info(
        "Class-D rewrites loaded: base=%d (#406) + extension=%d new (collisions=%d) "
        "= %d total from %s",
        len(base),
        n_added,
        n_collisions,
        len(merged),
        ext_p,
    )
    return merged


def assert_disjoint_q_train_q_test(q_train: list[str], q_test: list[str]) -> None:
    """Verify Q_train and Q_test share no exact-string questions.

    #406 Phase 0 already enforces this; we re-assert at Phase 1 launch as a
    defense-in-depth guard against drift in the HF data repo.

    Raises:
        AssertionError on any overlap.
    """
    overlap = set(q_train) & set(q_test)
    if overlap:
        raise AssertionError(
            f"Q_train ∩ Q_test contains {len(overlap)} question(s): {sorted(overlap)[:3]}..."
        )
