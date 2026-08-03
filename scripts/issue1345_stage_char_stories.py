#!/usr/bin/env python
"""Issue #1345 char-capture-ladders — thin per-cell kept-story stager (plan v13 §4 Phase C).

Stages ONE character cell's kept-story bundle from the HF data repo at the
plan §10 pinned revision into the extractor's expected layout:

    data/issue_1345/<variant>/stories/{kept_stories_<mode>_<model>.jsonl,
                                       story_yield_<mode>_<model>.json, ...}

Mechanics: ``hub.stage_hub_prefix`` (the canonical #1402 retried scoped-prefix
helper) mirrors the prefix VERBATIM under a same-filesystem scratch dir; this
script then applies the pure hub-rel -> local-rel mapping (strip the prefix)
and ``os.replace``s each file into ``<dest>/stories/`` — the
artifact-reuse.md (h)(iv) staged-layout contract: the consumer
(``issue1345_extract_turnstore.py --stories-dir``) opens the flat layout, and
the stage FAILS LOUD when the expected kept/yield files are missing.

Resume: stories mode skips when both expected files already exist under the
dest; turnstore mode skips on a ``.staged_complete`` sentinel written AFTER
the move loop (completeness-based, not presence-based — a crash inside the
rename loop must not read as "already staged"; r1 review Minor 1). Delete
the dir to re-stage. This script deliberately
does NOT import ``issue1345_common`` — that module compiles the
character-name regex from ``EPM_STORY_CHARACTER_NAME`` at import time, and a
stager must be safe to call for all 16 variants from one process.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Plan v13 §10 row 1: the kept-story bundle pin (16 cells, 33,806 stories).
STORIES_PIN = "704cc6cbc3f498cd4af3648c7055784ef71c905c"
# The #1034 proactive-overflow breadcrumb `upload_dir_sharded` commits at the
# CANONICAL prefix when the shard set was rerouted to the private overflow
# repo (schema: {overflow_repo, path_in_repo, ts, ...}).
OVERFLOW_POINTER_BASENAME = "OVERFLOW_POINTER.json"

CHAR_VARIANTS = tuple(
    f"char_{ch}{suf}"
    for ch in ("helios", "wren", "dana", "vex")
    for suf in ("", "_op", "_base", "_op_base")
)


def variant_mode_model(variant: str) -> tuple[str, str]:
    """(mode_slug, model_key) the gen/extract phases key their filenames on."""
    mode = "paired_op" if "_op" in variant else "paired"
    model = "pretrained" if variant.endswith("_base") else "instruct"
    return mode, model


def expected_files(variant: str) -> tuple[str, str]:
    """The two files the extractor opens (fail-loud entry check of the stage)."""
    mode, model = variant_mode_model(variant)
    return (f"kept_stories_{mode}_{model}.jsonl", f"story_yield_{mode}_{model}.json")


def _follow_overflow_pointer(
    mirror_root: Path, *, prefix: str, variant: str, scratch_of: Path
) -> tuple[Path, str] | None:
    """Follow a #1034 ``OVERFLOW_POINTER.json`` breadcrumb mirrored from the
    canonical prefix; returns ``(overflow_mirror_root, overflow_repo)`` or
    None when no pointer is present.

    The issue841 pointer-follow pattern (``issue841_scaling_common.
    _overflow_repo_for_bucket`` / ``fetch_capture_from_hf``), rebuilt on the
    canonical retried helper: detection reads the pointer file the canonical
    mirror ALREADY staged (zero extra Hub calls — never a bare
    ``list_repo_files`` full listing on the ~1M-file data repo, #833), then
    re-stages the SAME ``path_in_repo`` prefix from the PRIVATE overflow repo.
    ``upload_dir_sharded`` reroutes shards to the overflow repo with
    ``repo_type="model"`` (upload_sharded.py `(DEFAULT_OVERFLOW_REPO,
    "model")` routing), so the re-stage MUST use the model repo_type; auth
    rides the ambient HF token. Pointer-routed shards live at the overflow
    repo's DEFAULT branch — ``revision=None`` here by design; the caller's
    pinned-revision kwarg applies only to canonical-repo reads. Fail-loud: a
    malformed pointer (missing ``overflow_repo``) raises KeyError, a
    prefix-drifted pointer raises AssertionError, and a pointer naming an
    overflow prefix with no files raises FileNotFoundError from
    ``stage_hub_prefix`` (the issue841 "pointer says overflow but the shard
    is not there" contract).
    """
    pointer_path = mirror_root / OVERFLOW_POINTER_BASENAME
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text())
    overflow_repo = pointer["overflow_repo"]
    ptr_prefix = str(pointer.get("path_in_repo", ""))
    assert ptr_prefix == prefix, (
        f"overflow pointer prefix drift for {variant}: pointer says {ptr_prefix!r}, "
        f"stager expects {prefix!r}"
    )
    print(
        f"[stage] {variant} turnstore: OVERFLOW_POINTER at canonical prefix — "
        f"re-staging shard set from {overflow_repo} (private model repo, default branch)",
        flush=True,
    )
    if scratch_of.exists():
        shutil.rmtree(scratch_of)
    hub.stage_hub_prefix(overflow_repo, prefix, scratch_of, repo_type="model", revision=None)
    of_root = scratch_of / prefix
    assert of_root.is_dir(), (
        f"stage_hub_prefix mirrored nothing under {of_root} — overflow mirror drift"
    )
    return of_root, overflow_repo


def stage_variant_turnstore(
    variant: str,
    *,
    revision: str | None = None,
    dest_root: Path,
    repo_id: str = HF_DATA_REPO,
) -> Path:
    """Stage one cell's CAPTURED turnstore into ``<dest_root>/<variant>_turnstore``.

    The fits-phase sibling of :func:`stage_variant` (plan v13 §4 Phase F item
    3): the fill script's ``load_regime_xy`` opens the FLAT
    ``<variant>_turnstore`` subdir, while ``stage_hub_prefix`` mirrors the hub
    prefix verbatim — this applies the same strip-the-prefix mapping so the
    consumer layout is produced fail-loud (the artifact-reuse.md (h)(iv)
    staged-layout contract; #928/#1481). ``revision=None`` resolves one
    commit at call time (the capture job creates these stems, so no code-time
    pin exists; the shard sidecars carry the capture commit). OVERFLOW-AWARE
    (#1034 proactive path): when the canonical prefix carries an
    ``OVERFLOW_POINTER.json`` breadcrumb (the capture job's shards were
    rerouted to the private overflow repo over the public-storage ceiling),
    the shard set is re-staged from ``pointer["overflow_repo"]`` at the SAME
    prefix and merged into the SAME flat consumer layout — byte-identical to
    the canonical-path case (the pointer file itself is routing metadata and
    is excluded). NOTE: pointer-routed shards live at the overflow repo's
    DEFAULT branch; the ``revision`` kwarg applies only to the canonical-repo
    read. Idempotent:
    a ``.staged_complete`` sentinel (written only after the whole move loop
    + the pt-shard assert succeeded) -> skip; shard PRESENCE alone is not a
    skip key — a crash inside the rename loop leaves a partial dir a
    presence glob would wrongly skip (r1 review Minor 1).
    """
    assert variant in CHAR_VARIANTS, f"unknown character variant {variant!r}"
    dest = dest_root / f"{variant}_turnstore"
    sentinel = dest / ".staged_complete"
    if sentinel.is_file() and any(dest.glob("*_shard*.pt")):
        print(f"[stage] {variant} turnstore: already staged at {dest} — skipped", flush=True)
        return dest
    if dest.is_dir() and any(dest.iterdir()):
        # Partial stage (files but no completion sentinel): re-stage from
        # scratch so the consumer can never open a silently-shrunk shard set.
        print(f"[stage] {variant} turnstore: partial dir (no sentinel) — re-staging", flush=True)
        shutil.rmtree(dest)
    prefix = f"issue1345_framing/{variant}/analysis_tensors/turnstore"
    dest.mkdir(parents=True, exist_ok=True)
    scratch = dest.parent / f".hfstage_ts_{variant}"
    if scratch.exists():
        shutil.rmtree(scratch)
    staged = hub.stage_hub_prefix(repo_id, prefix, scratch, revision=revision)
    mirror_root = scratch / prefix
    assert mirror_root.is_dir(), (
        f"stage_hub_prefix mirrored nothing under {mirror_root} — prefix mirror drift"
    )
    # Overflow-routed cell (#1034): the canonical mirror may hold ONLY the
    # pointer breadcrumb (proactive path), or a pointer + a partial shard set
    # (reactive mid-store reroute) — either way the pointer names the private
    # repo holding the rest; merge BOTH mirrors into the flat consumer layout.
    scratch_of = dest.parent / f".hfstage_ts_of_{variant}"
    overflow = _follow_overflow_pointer(
        mirror_root, prefix=prefix, variant=variant, scratch_of=scratch_of
    )
    mirror_roots = [mirror_root]
    overflow_repo: str | None = None
    if overflow is not None:
        of_root, overflow_repo = overflow
        mirror_roots.append(of_root)
    n_moved = 0
    for root in mirror_roots:
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.name == OVERFLOW_POINTER_BASENAME:
                # The breadcrumb is routing metadata, not part of the consumer
                # bundle — the staged layout stays byte-identical to the
                # canonical-path case.
                continue
            out = dest / f.relative_to(root)
            out.parent.mkdir(parents=True, exist_ok=True)
            os.replace(f, out)
            n_moved += 1
    shutil.rmtree(scratch)
    if scratch_of.exists():
        shutil.rmtree(scratch_of)
    pts = sorted(dest.glob("*_shard*.pt"))
    assert pts, (
        f"staged turnstore for {variant} has no *_shard*.pt (moved {n_moved} files from "
        f"{repo_id}@{revision or 'resolved-head'}:{prefix}"
        + (f" + overflow {overflow_repo}" if overflow_repo else "")
        + ") — consumer layout violated"
    )
    # Completion sentinel LAST (atomic tmp+replace): the resume predicate keys
    # on it, so a crash anywhere above re-stages instead of silently skipping.
    tmp_sentinel = dest / ".staged_complete.tmp"
    tmp_sentinel.write_text(
        json.dumps(
            {
                "variant": variant,
                "n_files": n_moved,
                "n_pt_shards": len(pts),
                "revision": revision or "resolved-head",
                "overflow_repo": overflow_repo,
                "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    )
    os.replace(tmp_sentinel, sentinel)
    print(
        f"[stage] {variant} turnstore: {n_moved} files ({len(staged)} listed"
        + (f" canonical + overflow {overflow_repo}" if overflow_repo else "")
        + f", {len(pts)} pt shards) -> {dest}",
        flush=True,
    )
    return dest


def stage_variant(
    variant: str,
    *,
    revision: str = STORIES_PIN,
    dest_root: Path = Path("data/issue_1345"),
    repo_id: str = HF_DATA_REPO,
) -> Path:
    """Stage one cell's stories bundle; returns the stories dir. Idempotent."""
    assert variant in CHAR_VARIANTS, f"unknown character variant {variant!r}"
    dest = dest_root / variant / "stories"
    kept_name, yield_name = expected_files(variant)
    if (dest / kept_name).is_file() and (dest / yield_name).is_file():
        print(f"[stage] {variant}: already staged at {dest} — skipped", flush=True)
        return dest

    prefix = f"issue1345_framing/{variant}/raw_completions/stories"
    dest.mkdir(parents=True, exist_ok=True)
    # Scratch INSIDE the dest parent: same filesystem, so os.replace is atomic
    # (never a cross-device rename — the #1335 EXDEV trap).
    scratch = dest.parent / f".hfstage_{variant}"
    if scratch.exists():
        shutil.rmtree(scratch)
    staged = hub.stage_hub_prefix(repo_id, prefix, scratch, revision=revision)
    mirror_root = scratch / prefix
    assert mirror_root.is_dir(), (
        f"stage_hub_prefix mirrored nothing under {mirror_root} — prefix mirror drift"
    )
    n_moved = 0
    for f in sorted(mirror_root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(mirror_root)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(f, out)
        n_moved += 1
    shutil.rmtree(scratch)
    # Fail-loud entry check: the consumer's own open targets must exist.
    for name in (kept_name, yield_name):
        assert (dest / name).is_file(), (
            f"staged bundle for {variant} lacks {name} (staged {n_moved} files from "
            f"{repo_id}@{revision}:{prefix}) — consumer layout violated"
        )
    print(
        f"[stage] {variant}: {n_moved} files ({len(staged)} listed) from "
        f"{repo_id}@{revision[:12]}:{prefix} -> {dest}",
        flush=True,
    )
    return dest


def main() -> None:
    """CLI wrapper: stage one variant (or --all 16) at the pinned revision."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--variant", choices=CHAR_VARIANTS, help="one character cell")
    ap.add_argument("--all", action="store_true", help="stage all 16 cells")
    ap.add_argument(
        "--kind",
        choices=("stories", "turnstore"),
        default="stories",
        help="stories = kept-story bundle at the plan pin (capture input); "
        "turnstore = captured activation shards into <dest-root>/<variant>_turnstore "
        "(fits input; default revision = resolved head at call time)",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help=f"HF revision (stories default: the plan pin {STORIES_PIN[:12]}; "
        "turnstore default: resolved head)",
    )
    ap.add_argument(
        "--dest-root",
        type=Path,
        default=None,
        help="stories default: data/issue_1345; REQUIRED for --kind turnstore "
        "(the fits --stage-root, e.g. /mnt/eps-data/$USER/issue1887_lambda_audit/issue1345)",
    )
    ap.add_argument("--repo", default=HF_DATA_REPO)
    args = ap.parse_args()
    assert args.all or args.variant, "pass --variant <cell> or --all"
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (pod .env / lane metadata env)"
    variants = CHAR_VARIANTS if args.all else (args.variant,)
    for v in variants:
        if args.kind == "turnstore":
            assert args.dest_root is not None, "--kind turnstore requires --dest-root"
            stage_variant_turnstore(
                v, revision=args.revision, dest_root=args.dest_root, repo_id=args.repo
            )
        else:
            stage_variant(
                v,
                revision=args.revision or STORIES_PIN,
                dest_root=args.dest_root or Path("data/issue_1345"),
                repo_id=args.repo,
            )


if __name__ == "__main__":
    main()
