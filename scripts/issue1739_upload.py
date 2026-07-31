"""Durability CLI for issue #1739 (round-2 C1): HF uploads + result git push.

Stages (each idempotent; the dispatcher sequences them):

- ``--stage raw``: PACK-FIRST (round 4). The generation module writes ONE
  JSON per (context, seed) (~255k files across 3 behaviors), so a naive
  ``upload_folder`` of the raw tree trips the Hub's 10k-files-per-directory
  commit cap (``hub.HubDirFileCountError`` staging 115,941 files into
  ``.../labeling/hallucination`` — the production crash this round fixes).
  The stage now packs the tree into <= 9 MB ``<group>.shardNN.jsonl``
  line-shards + a ``pack_manifest.json`` (``scripts/issue1739_pack.py``;
  idempotent census-keyed re-pack), then makes ONE bulk ``upload_folder``
  commit of the tiny shard set to ``issue1739_ctxmap/raw_completions/`` —
  BEFORE any scoring/judging (plan Step 2a-bis upload-before-scoring;
  Upload Policy raw-completions row). Never a per-file loop (the #664
  504-storm class).
- ``--stage tensors``: ONE bulk commit of the analysis-tensors tree (r_B
  direction npz, map diagnostics side-files, per-cell prediction sidecars)
  to ``issue1739_ctxmap/analysis_tensors/`` (plan §10 destinations; #521
  downstream-input tensors upload before pod termination).
- ``--stage results-git``: commit ``eval_results/issue_1739`` +
  ``figures/issue_1739`` by explicit path on the issue branch, push, verify
  via ``rev-list --count origin/<branch>..HEAD == 0`` (retry once), then
  assert every DECLARED result file is present in the PUSHED tree via
  ``git ls-tree`` (pod-side-reporting.md § Result-push + artifact-presence
  assert #1325). Exits non-zero on any miss — never "done" with an
  unpushed/uncommitted result.

Uploads verify with a scoped ``verify_repo_paths_uploaded`` exact-set
listing. ``--dry-run`` enumerates + logs without touching the Hub/git (the
smoke path; smoke artifacts never upload to canonical prefixes and are
never git-committed).

Plan-glob parity (#825): the two HF stages carry NO eligibility filter —
the WHOLE tree under each root uploads (every plan-declared class under
those roots is eligible by construction).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue1739_upload.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

HF_PREFIX = "issue1739_ctxmap"
BRANCH = "issue-1739"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=("raw", "tensors", "results-git"))
    ap.add_argument("--raw-root", type=Path, default=Path("raw_completions/issue_1739"))
    ap.add_argument(
        "--pack-root",
        type=Path,
        default=None,
        help="shard output dir for --stage raw (default: <raw-root>_packed beside it)",
    )
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument("--percell-glob-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument("--results-root", type=Path, default=Path("eval_results/issue_1739"))
    ap.add_argument("--figures-root", type=Path, default=Path("figures/issue_1739"))
    ap.add_argument("--branch", default=BRANCH)
    # UPLOAD_PREFIX_EXEMPT: issue-1739-DEDICATED uploader (issue1739_*.py, never a
    # shared fit script); a child issue reusing it copies + renames the script and
    # its prefix constant with it, so the #1005 silent-inherit clobber cannot fire.
    ap.add_argument("--hf-prefix", default=HF_PREFIX)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate + log only (the smoke path; no Hub/git mutation)",
    )
    return ap.parse_args(argv)


def _enumerate(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def upload_tree(local_root: Path, path_in_repo: str, *, dry_run: bool, what: str) -> list[str]:
    """One bulk upload_folder commit + scoped exact-set verify; fail loud."""
    files = _enumerate(local_root)
    if not files:
        raise SystemExit(f"[upload] {what}: nothing to upload under {local_root}")
    rel = [str(p.relative_to(local_root)) for p in files]
    print(f"[upload] {what}: {len(files)} files under {local_root} -> {path_in_repo}", flush=True)
    if dry_run:
        for r in rel[:10]:
            print(f"[upload]   (dry-run) {path_in_repo}/{r}", flush=True)
        return rel
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        local_root,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        path_in_repo=path_in_repo,
    )
    if not url:
        raise SystemExit(f"[upload] {what}: upload returned no path (FAILED) — not proceeding")
    expected = [f"{path_in_repo}/{r}" for r in rel]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), hub.DEFAULT_DATASET_REPO, expected, path_in_repo=path_in_repo
    )
    if missing:
        raise SystemExit(
            f"[upload] {what}: {len(missing)} files missing on the Hub after upload "
            f"(first: {missing[:5]})"
        )
    print(f"[upload] {what}: verified {len(expected)} files on the Hub", flush=True)
    return rel


def raw_stage(args: argparse.Namespace) -> int:
    """Pack-first raw-completions upload (#1739 round 4).

    Packs the per-(context, seed) JSON tree into <= 9 MB jsonl line-shards
    (idempotent — census-matched groups are reused), then uploads the shard
    set. MECHANISM CHOICE (the two remedies the HubDirFileCountError guard
    names): ONE bulk ``upload_folder`` commit via :func:`upload_tree`, NOT
    ``orchestrate.upload_sharded.upload_dir_sharded`` — after packing the
    file count is tiny (~hundreds of shards + manifest, far under the
    10k/dir server cap and the 2k advisory throughput tier) and the total
    is non-LFS text, so one commit + one scoped exact-set verify is the
    cheapest correct shape; ``upload_dir_sharded`` commits ONE FILE PER
    COMMIT (against the fleet-shared 256 commits/hr budget) with
    delete-local semantics aimed at bigger-than-disk tensor stores — the
    wrong tool at this scale. Exact-set verify semantics are unchanged:
    :func:`upload_tree` verifies the shard names + manifest on the Hub.
    """
    raw_root: Path = args.raw_root
    pack_root: Path = args.pack_root or raw_root.parent / f"{raw_root.name}_packed"
    if args.dry_run:
        from scripts.issue1739_pack import group_files

        groups = group_files(raw_root)
        total = sum(len(g["files"]) for g in groups.values())
        print(
            f"[upload] raw: (dry-run) {total} per-context JSONs in {len(groups)} group(s) "
            f"would pack to {pack_root} then upload -> {args.hf_prefix}/raw_completions",
            flush=True,
        )
        for key in sorted(groups):
            print(f"[upload]   (dry-run) group {key}: {len(groups[key]['files'])} files")
        return 0
    from scripts.issue1739_pack import pack_raw_tree

    pack_raw_tree(raw_root, pack_root)
    # UPLOAD_PREFIX_EXEMPT: issue-1739-dedicated uploader; child issues copy+rename the script (and its prefix constant) per house convention
    upload_tree(
        pack_root,
        f"{args.hf_prefix}/raw_completions",
        dry_run=False,
        what="raw completions (packed jsonl line-shards + manifest)",
    )
    return 0


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args], capture_output=True, text=True, check=check
    )


def declared_result_paths(results_root: Path, figures_root: Path) -> list[str]:
    """The round's DECLARED git-destined result files (plan §6.5 realized).

    Per-behavior fits summaries + dv_dataset JSONs + every figure file the
    figures phase produced. Only files that EXIST enter the assert set (a
    behavior that legitimately did not run this invocation is not a miss);
    an EMPTY set on a results-git call is a failure (nothing to land means
    the phase should not have been reached).
    """
    declared: list[str] = []
    declared += [str(p) for p in sorted(results_root.glob("*/arm_results/all_arms_spearman.json"))]
    declared += [str(p) for p in sorted(results_root.glob("dv_dataset/*/labeling.json"))]
    declared += [str(p) for p in sorted(results_root.glob("*/map_diagnostics.json"))]
    if figures_root.exists():
        declared += [
            str(p) for p in _enumerate(figures_root) if p.suffix in (".png", ".pdf", ".json")
        ]
    return declared


def results_git_stage(args: argparse.Namespace) -> int:
    declared = declared_result_paths(args.results_root, args.figures_root)
    if not declared:
        raise SystemExit(
            f"[upload] results-git: ZERO declared result files under {args.results_root} — "
            "refusing to declare done"
        )
    print(f"[upload] results-git: {len(declared)} declared result files", flush=True)
    if args.dry_run:
        for p in declared[:10]:
            print(f"[upload]   (dry-run) would commit {p}", flush=True)
        return 0
    # Stage by explicit path (never add -A). eval_results JSONs + figures only.
    _git(["add", "--", str(args.results_root), str(args.figures_root)], check=False)
    # Staged-index verification: gitignored files silently skipped by a
    # dir-path add (the #958 class) get force-added by explicit path.
    skipped = _git(
        [
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            str(args.results_root),
            str(args.figures_root),
        ],
        check=False,
    ).stdout.splitlines()
    convention = [p for p in skipped if p.endswith((".json", ".jsonl", ".png", ".pdf"))]
    if convention:
        _git(["add", "-f", "--", *convention])
        print(f"[upload] results-git: force-added {len(convention)} gitignore-skipped files")
    diff = _git(["diff", "--cached", "--name-only"], check=False).stdout.strip()
    if diff:
        _git(
            [
                "commit",
                "-m",
                "task #1739: eval results + figures (dispatcher results phase)",
            ]
        )
    else:
        print("[upload] results-git: nothing new to commit (idempotent re-run)", flush=True)
    # Push + rev-list verify (retry once) — pod-side-reporting.md § Result-push.
    for attempt in (1, 2):
        push = _git(["push", "origin", f"HEAD:{args.branch}"], check=False)
        behind = _git(
            ["rev-list", "--count", f"origin/{args.branch}..HEAD"], check=False
        ).stdout.strip()
        if push.returncode == 0 and behind == "0":
            break
        print(
            f"[upload] results-git: push attempt {attempt} rc={push.returncode} "
            f"behind={behind!r} stderr={push.stderr.strip()[:300]}",
            flush=True,
        )
        if attempt == 2:
            raise SystemExit("[upload] results-git: push verification FAILED after retry")
        _git(["fetch", "origin", args.branch], check=False)
    # Artifact-presence assert (#1325): every declared file in the PUSHED tree.
    missing = [
        p
        for p in declared
        if not _git(
            ["ls-tree", "-r", f"origin/{args.branch}", "--name-only", "--", p], check=False
        ).stdout.strip()
    ]
    if missing:
        raise SystemExit(
            f"[upload] results-git: {len(missing)} declared result files MISSING from the "
            f"pushed tree (first: {missing[:5]})"
        )
    print(
        f"[upload] results-git: push verified; {len(declared)} declared files present in "
        f"origin/{args.branch}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    args = _parse_args(argv)
    if args.stage == "raw":
        return raw_stage(args)
    if args.stage == "tensors":
        # UPLOAD_PREFIX_EXEMPT: issue-1739-dedicated uploader; child issues copy+rename the script (and its prefix constant) per house convention
        upload_tree(
            args.tensors_root,
            f"{args.hf_prefix}/analysis_tensors",
            dry_run=args.dry_run,
            what="analysis tensors (r_B + diagnostics)",
        )
        preds = sorted(args.percell_glob_root.glob("*/arm_results/percell/preds/*.npz"))
        if preds:
            # percell prediction sidecars live under eval_results (gitignored
            # npz) — they ride the SAME analysis-tensors prefix on the Hub.
            for behavior_dir in sorted({p.parents[3] for p in preds}):
                # UPLOAD_PREFIX_EXEMPT: issue-1739-dedicated uploader; child issues copy+rename the script (and its prefix constant) per house convention
                upload_tree(
                    behavior_dir / "arm_results" / "percell" / "preds",
                    f"{args.hf_prefix}/analysis_tensors/percell_preds/{behavior_dir.name}",
                    dry_run=args.dry_run,
                    what=f"per-cell prediction sidecars ({behavior_dir.name})",
                )
        else:
            print("[upload] tensors: no percell prediction sidecars found (ok pre-fits)")
        return 0
    return results_git_stage(args)


if __name__ == "__main__":
    sys.exit(main())
