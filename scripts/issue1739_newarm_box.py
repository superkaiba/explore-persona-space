"""new-arm-round per-box helper (task #1739, plan v8): stage-meta + self-upload.

Two subcommands shared by all four ``issue1739_newarm_*.sh`` drivers:

- ``stage-meta``: record the RESOLVED data-repo commit sha for the box's
  unpinned ``issue1739_ctxmap/*`` reads (plan v8 §4 collect requirements —
  "stage-time sha recording") into a small meta JSON under the leg's out-root,
  so the VM collect phase can state exactly which repo state every leg
  consumed.
- ``upload``: fail-loud per-box HF self-upload (the gap2 pattern hardened):
  each ``--pairs LOCAL_DIR:REPO_PREFIX`` uploads as ONE ``upload_folder``
  commit (never a per-file loop — the #664 504-storm class) wrapped in
  ``hub.retry_transient``, then EXACT-SET verified against a scoped
  server-side listing (``hub.verify_repo_paths_uploaded``) — the plan's
  ``self_upload`` sentinel contract ("fail-loud upload_folder + exact-set
  verify"). Any missing file raises.

Counts-only logging; no corpus content printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

DATA_REPO = "superkaiba1/explore-persona-space-data"


def _git_commit() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, check=False
    )
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def resolve_data_repo_main_sha() -> str:
    """Resolved target commit of the data repo's ``main`` ref (retried)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    refs = hub.retry_transient(
        lambda: HfApi().list_repo_refs(DATA_REPO, repo_type="dataset"),
        what=f"list_repo_refs {DATA_REPO}",
    )
    for br in refs.branches:
        if br.name == "main":
            return str(br.target_commit)
    raise RuntimeError(f"data repo {DATA_REPO} has no 'main' branch ref")


def cmd_stage_meta(args: argparse.Namespace) -> int:
    sha = resolve_data_repo_main_sha()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "leg": args.leg,
        "behavior": args.behavior,
        "data_repo": DATA_REPO,
        "data_repo_main_sha_at_stage": sha,
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)
    print(f"[newarm-box] stage-meta: {DATA_REPO}@{sha[:12]} -> {out}", flush=True)
    return 0


def _upload_one(local: Path, dest: str) -> int:
    """One dir -> one bulk upload_folder commit + exact-set verify. Returns n files."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    if not local.is_dir():
        raise RuntimeError(f"[newarm-box] upload pair local path is not a directory: {local}")
    rel_files = sorted(p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file())
    if not rel_files:
        raise RuntimeError(f"[newarm-box] nothing to upload under {local}")
    # 10k-files-per-dir Hub commit cap (#1190) — guard BEFORE the retry wrapper
    # (a guard raise is deterministic; retrying it burns budget for nothing).
    hub.assert_hub_dir_filecounts(local, dest)
    api = HfApi()
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(local),
            path_in_repo=dest,
            repo_id=DATA_REPO,
            repo_type="dataset",
        ),
        what=f"upload_folder {local} -> {dest}",
    )
    expected = [f"{dest}/{rel}" for rel in rel_files]
    missing = hub.verify_repo_paths_uploaded(
        api, DATA_REPO, expected, path_in_repo=dest, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(
            f"[newarm-box] exact-set verify FAILED for {dest}: {len(missing)}/{len(expected)} "
            f"missing (first: {missing[:5]})"
        )
    print(f"[newarm-box] upload OK: {local} -> {dest} ({len(expected)} files verified)", flush=True)
    return len(expected)


def cmd_upload(args: argparse.Namespace) -> int:
    total = 0
    for pair in args.pairs:
        local_s, _, dest = pair.partition(":")
        if not dest:
            raise SystemExit(f"--pairs must be LOCAL_DIR:REPO_PREFIX, got {pair!r}")
        total += _upload_one(Path(local_s), dest.strip("/"))
    print(f"[newarm-box] self-upload complete: {total} files across {len(args.pairs)} pair(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("stage-meta", help="record the resolved data-repo main sha")
    sm.add_argument("--leg", required=True, help="leg label (e.g. fc/evil, oracle/sycophancy)")
    sm.add_argument("--behavior", default=None)
    sm.add_argument("--out", required=True, help="meta JSON path (under the leg out-root)")
    up = sub.add_parser("upload", help="fail-loud bulk self-upload + exact-set verify")
    up.add_argument(
        "--pairs",
        action="append",
        required=True,
        metavar="LOCAL_DIR:REPO_PREFIX",
        help="repeatable; REPO_PREFIX is the full literal dest prefix under the data repo "
        "(e.g. issue1739_new_arm_round/fc/evil)",
    )
    args = ap.parse_args(argv)
    rc = cmd_stage_meta(args) if args.cmd == "stage-meta" else cmd_upload(args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize teardown


if __name__ == "__main__":
    main()
