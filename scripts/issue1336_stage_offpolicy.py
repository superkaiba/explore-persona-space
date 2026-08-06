#!/usr/bin/env python3
"""Phase EXT_off staging helper — bring the 20 off-diagonal
(activation-checkpoint i × text-source j, i != j) cells to the pod's
``${WORKLOAD_ROOT}/data/generations_offpolicy/<slug_j>/<corpus>/answers.jsonl``
layout (plan v15 §4 EXT_off + §4 reuse-check (h) leg (iv)).

Layout contract: **verbatim mirror** of the diagonal on-policy generation
layout `data/issue_1336/gen{,_v2}/<slug>/<corpus>/answers.jsonl` — no new
staging transformation (plan §4 "reuses the round-3 verbatim-mirror
layout"). ONE consumer (the off-policy capture rig) opens the staged
tree; the map from HF prefix to local relative path is identity.

Sources:
  - wave-1 generation shards @ ``cm.WAVE1_HF_REV = 8c54f9fc`` for the
    three wave-1 corpora (lmsys5k -> slug_j==base/sft/dpo/rlvr/rlvr_long
    at layout `gen/<slug_j>/lmsys5k/answers.jsonl`; gsm8k_train5k;
    gsm8k_test1319) — the reused-verbatim wave-1 pins from cm.
  - round-3 v2 generation shards @ the round-3-final data-repo revision
    (resolved at invocation via HfApi().repo_info) for the extended
    corpora (lmsys23k, gsm8k_train_full, math7500, if11k, uf11k, sft11k).
    The pin used for staging is recorded in the output manifest so the
    consumer sees exactly which shards were staged.

For extended corpora the on-disk root is `data/issue_1336/gen_v2/<slug_j>/<slug>/`.
For wave-1 corpora the on-disk root is `data/issue_1336/gen/<slug_j>/<stem>/`
where ``stem`` matches the wave-1 generation naming (lmsys5k / gsm8k_train5k /
gsm8k_test1319).

Manifest-aware staging: the answers.jsonl file MAY have been sharded on
upload (the ">9.5MB shard contract" enforced by
`issue1336_stage_corpora._maybe_line_split_for_upload`); the reader
ports the manifest reassembly recipe from
``scripts/issue1336_diagnose_g1.py::_maybe_reassemble_answers``.

Smoke (``--smoke``): stages a single tiny off-diagonal cell (defaults to
the sole SMOKE_CORPORA_V2 corpus on a SMOKE_MODELS x SMOKE_MODELS off-diag
pair) so the layout mirror is exercised end-to-end without pulling
production corpora into a local dev environment. Smoke never uploads.

Every ``__main__`` invocation exits explicitly to sidestep the
PyGILState_Release atexit race (gotchas.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402
from explore_persona_space.orchestrate import env  # noqa: E402

logger = logging.getLogger("issue1336_stage_offpolicy")

DATA_ROOT = _REPO_ROOT / "data" / "issue_1336"
GEN_ROOT = DATA_ROOT / "gen"  # wave-1 layout
GEN_V2_ROOT = DATA_ROOT / "gen_v2"  # round-3 v2 layout

OFFPOL_MANIFEST_NAME = "stage_offpolicy_manifest.json"
OFFPOL_MANIFEST_SUBDIR = "generations_offpolicy_stage"

MODELS_ALL: tuple[str, ...] = tuple(cm.MODELS.keys())  # base, sft, dpo, rlvr, rlvr_long


@dataclass
class StageContext:
    smoke: bool
    corpora: tuple[str, ...]
    models: tuple[str, ...]
    cells: tuple[tuple[str, str, str], ...]  # (model_i, model_j, corpus) — i != j
    v2_main_revision: str | None = None
    manifest_rows: list[dict[str, Any]] = field(default_factory=list)


def _hub_helpers():
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    return HfApi(), hf_hub_download, hub


def _resolve_v2_revision(api) -> str:
    from explore_persona_space.orchestrate import hub

    info = hub.retry_transient(
        lambda: api.repo_info(repo_id=cm.HF_DATA_REPO, repo_type="dataset", revision="main"),
        what=f"stage_offpolicy repo_info {cm.HF_DATA_REPO}@main",
    )
    sha = getattr(info, "sha", None)
    assert sha, f"repo_info returned no sha for {cm.HF_DATA_REPO}"
    return str(sha)


def _prefix_and_layout(model_j: str, corpus: str) -> tuple[str, str, Path]:
    """Return (hf_prefix, revision, local_dest_dir) for a text-source (j, corpus).

    Wave-1 corpora live under `raw_completions/generation/<model_j>/<stem>/`
    at `WAVE1_HF_REV`; v2 corpora under `raw_completions/generation_v2/<model_j>/<corpus>/`
    at the resolved main revision.
    """
    if corpus == "lmsys23k":
        subprefix, stem, root = "generation", "lmsys5k", GEN_ROOT
        revision = cm.WAVE1_HF_REV
    elif corpus == "gsm8k_train_full":
        subprefix, stem, root = "generation", "gsm8k_train5k", GEN_ROOT
        revision = cm.WAVE1_HF_REV
    elif corpus == "gsm8k_test1319":
        subprefix, stem, root = "generation", "gsm8k_test1319", GEN_ROOT
        revision = cm.WAVE1_HF_REV
    else:
        subprefix, stem, root = "generation_v2", corpus, GEN_V2_ROOT
        revision = "main"  # placeholder — resolved once per StageContext
    prefix = f"{cm.HF_PREFIX_1336}/raw_completions/{subprefix}/{model_j}/{stem}"
    dest = root / model_j / stem
    return prefix, revision, dest


def _stage_answers_files(
    api,
    hf_hub_download,
    hub,
    prefix: str,
    revision: str,
    dest: Path,
) -> tuple[Path, list[str]]:
    """Stage answers.jsonl (or its shard manifest+parts) for one prefix.

    Ports the sharded-answers manifest reader from
    ``scripts/issue1336_diagnose_g1.py`` (``_stage_prefix`` +
    ``_maybe_reassemble_answers``). Returns
    (final_answers_path, staged_relative_paths).
    """
    dest.mkdir(parents=True, exist_ok=True)
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: lambda is retried by hub.retry_transient
            api.list_repo_tree(
                cm.HF_DATA_REPO,
                path_in_repo=prefix,
                repo_type="dataset",
                revision=revision,
                recursive=True,
            )
        ),
        what=f"stage_offpolicy tree {prefix}@{revision}",
    )
    files = [e.path for e in entries if hasattr(e, "size")]
    assert files, (
        f"no files under {prefix} on {cm.HF_DATA_REPO}@{revision} — cannot stage off-policy cell"
    )

    staged: list[str] = []
    for rel in sorted(files):
        base = Path(rel).name
        # Only stage answers.jsonl + its shard manifest + shard parts. The
        # gen-phase shard contract is `answers.shard{NN}.jsonl` (per
        # `scripts/issue1336_gen_answers.py::_maybe_line_split_for_upload`),
        # NOT `answers.jsonl.part*`. Accept both spellings so any future
        # shard-naming change stays reasonably tolerant.
        if not (
            base == "answers.jsonl"
            or base == "answers.manifest.json"
            or base.startswith("answers.shard")
            or base.startswith("answers.jsonl.part")
        ):
            continue
        local_target = dest / base
        if local_target.exists() and local_target.stat().st_size > 0:
            staged.append(base)
            continue
        hub.retry_transient(
            lambda r=rel: hf_hub_download(
                repo_id=cm.HF_DATA_REPO,
                repo_type="dataset",
                filename=r,
                revision=revision,
                local_dir=dest,
            ),
            what=f"stage_offpolicy download {rel}",
        )
        mirrored = dest / rel
        if mirrored.exists() and mirrored != local_target:
            local_target.parent.mkdir(parents=True, exist_ok=True)
            mirrored.rename(local_target)
        staged.append(base)

    # Reassemble sharded answers.jsonl if the manifest exists and the single
    # file is absent (the diagnose_g1 recipe).
    single = dest / "answers.jsonl"
    manifest_path = dest / "answers.manifest.json"
    if not single.exists() and manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        tmp = dest / "answers.jsonl.tmp"
        h = hashlib.sha256()
        with tmp.open("wb") as out:
            for part in m["parts"]:
                data = (dest / part).read_bytes()
                h.update(data)
                out.write(data)
        assert h.hexdigest() == m["total_sha256"], (
            f"reassembled answers.jsonl sha mismatch under {dest} for {prefix}"
        )
        tmp.replace(single)
        logger.info(
            "[offpol] reassembled answers.jsonl from %d parts under %s", len(m["parts"]), dest
        )
    assert single.exists() and single.stat().st_size > 0, (
        f"answers.jsonl absent after staging under {dest} for {prefix}"
    )
    return single, staged


def _build_cells(
    models: tuple[str, ...], corpora: tuple[str, ...]
) -> tuple[tuple[str, str, str], ...]:
    """Cells: (activation-checkpoint i, text-source j, corpus) with i != j.

    Consumers stage the TEXT SOURCE j x corpus (the activation checkpoint i
    is applied on the pod at forward time). The cell tuple carries i for
    manifest-side auditing (so the reader can reconcile which off-diagonal
    pair a staged tree covers).
    """
    cells = []
    for i in models:
        for j in models:
            if i == j:
                continue
            for corpus in corpora:
                cells.append((i, j, corpus))
    return tuple(cells)


def _run_stage(ctx: StageContext) -> int:
    api, dl, hub = _hub_helpers()
    if any(_prefix_and_layout(m_j, c)[1] == "main" for _m_i, m_j, c in ctx.cells):
        # At least one v2 corpus is in the set — resolve the main sha once.
        ctx.v2_main_revision = _resolve_v2_revision(api)
        logger.info("[offpol] resolved v2 data-repo revision: %s", ctx.v2_main_revision)

    staged_paths_seen: set[str] = set()
    for i, j, corpus in ctx.cells:
        prefix, revision, dest = _prefix_and_layout(j, corpus)
        if revision == "main":
            assert ctx.v2_main_revision, "v2_main_revision unresolved"
            revision = ctx.v2_main_revision
        # Idempotent per (j, corpus) — many cells share the same TEXT source.
        key = f"{j}::{corpus}"
        if key in staged_paths_seen:
            row = {
                "activation_checkpoint_i": i,
                "text_source_j": j,
                "corpus": corpus,
                "prefix": prefix,
                "revision": revision,
                "dest": str(dest.relative_to(_REPO_ROOT)),
                "skipped_reuse": True,
            }
            ctx.manifest_rows.append(row)
            continue
        answers_path, staged_files = _stage_answers_files(api, dl, hub, prefix, revision, dest)
        staged_paths_seen.add(key)
        row = {
            "activation_checkpoint_i": i,
            "text_source_j": j,
            "corpus": corpus,
            "prefix": prefix,
            "revision": revision,
            "dest": str(dest.relative_to(_REPO_ROOT)),
            "answers_path": str(answers_path.relative_to(_REPO_ROOT)),
            "staged_files": staged_files,
        }
        ctx.manifest_rows.append(row)
        logger.info(
            "[offpol] staged (i=%s, j=%s, %s) -> %s (%d files)",
            i,
            j,
            corpus,
            dest,
            len(staged_files),
        )
    return 0


def _write_manifest(ctx: StageContext) -> Path:
    root = DATA_ROOT / OFFPOL_MANIFEST_SUBDIR
    if ctx.smoke:
        root = DATA_ROOT / (OFFPOL_MANIFEST_SUBDIR + "_smoke")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "plan_version": "v15",
        "phase": "ext_off_stage",
        "smoke": ctx.smoke,
        "models": list(ctx.models),
        "corpora": list(ctx.corpora),
        "wave1_hf_rev": cm.WAVE1_HF_REV,
        "v2_main_revision": ctx.v2_main_revision,
        "n_cells": len(ctx.cells),
        "cells": ctx.manifest_rows,
        "generated_ts": int(time.time()),
    }
    path = root / OFFPOL_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("[offpol] wrote manifest %s (%d cells)", path, len(ctx.manifest_rows))
    return path


def _build_context(args) -> StageContext:
    if args.smoke:
        # Smoke: ONE off-diagonal cell — the smallest possible exercise of
        # the verbatim-mirror layout leg. i=dpo/j=rlvr on the sole
        # SMOKE_CORPORA_V2 corpus (lmsys23k -> wave-1 lmsys5k prefix).
        # Restricts fetches to a single (j, corpus) pair, so the smoke
        # tolerates the HF path staying reachable but adds no additional
        # cells beyond what's needed to prove the layout mirror.
        corpora = tuple(cm.SMOKE_CORPORA_V2)
        models = ("dpo", "rlvr")  # picks ONE off-diagonal pair
    else:
        corpora = tuple(cm.V2_CORPORA.keys())
        models = MODELS_ALL
    cells = _build_cells(models, corpora)
    return StageContext(smoke=args.smoke, corpora=corpora, models=models, cells=cells)


def run(args) -> int:
    env.load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    ctx = _build_context(args)
    logger.info(
        "[offpol] stage start smoke=%s models=%s corpora=%s n_cells=%d",
        ctx.smoke,
        list(ctx.models),
        list(ctx.corpora),
        len(ctx.cells),
    )
    _run_stage(ctx)
    _write_manifest(ctx)
    print(f"[offpol] stage complete: {len(ctx.manifest_rows)} rows", flush=True)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="issue1336 off-policy staging (Phase EXT_off)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--full", action="store_true", help="stage all 20 off-diagonal cells (default)"
    )
    mode.add_argument("--smoke", action="store_true", help="stage a tiny off-diagonal slice")
    args = ap.parse_args(argv)
    if not args.smoke and not args.full:
        args.full = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
