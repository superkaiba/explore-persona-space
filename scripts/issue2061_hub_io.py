"""Cross-machine HF staging + upload wiring for task #2061 (plan §9 `off_pod_phases`).

Every cross-machine dependency of the #2061 pipeline runs through the HF data
repo (plan §9 "Off-pod phase declaration"): P1 uploads `sae_encoded/` before
its GPU pod terminates; P2 (cpu-bigmem) stages P1's encoded targets + #1336
turnstore shards and uploads `analysis_tensors/per_feature_r2/`; P3 (8x GPU)
stages those plus P2's R² files and uploads `analysis_tensors/null/`; P4
uploads `analysis_tensors/fitness/` before its ephemeral `eval` pod terminates
(declared v6 — the #1738 fit-summary-JSON loss class); P5 (VM) fetches the
P2/P3/P4 outputs. This module is the ONE implementation of those legs — the
phase scripts + `issue2061_dispatch.sh` call it, never bare hub APIs.

All Hub traffic rides the canonical retried helpers (`hub.stage_hub_file` /
`hub.stage_hub_prefix` / `upload_sharded.upload_dir_sharded` — #1402/#1335;
never a bare `hf_hub_download`/`upload_folder`). Turnstore staging maps
hub-relative shard paths to the consumer's OWN layout
(`<root>/turnstore_<stage>_<render>_<corpus>/<shard>.pt`) via an explicit
hub-rel -> local-rel map (the #1774 `stage_hub_prefix` dest-is-mirror-root
trap does not apply to it).

CLI contract: machine-readable output (staged dir paths) goes to STDOUT;
all logging goes to STDERR — `$(... stage ...)` captures ONLY the path.

Usage:
    uv run python scripts/issue2061_hub_io.py upload --what fitness \
        --dir eval_results/issue_2061/fitness
    uv run python scripts/issue2061_hub_io.py stage --what per-feature-r2 \
        --root data/issue_2061/hf_dl
    uv run python scripts/issue2061_hub_io.py stage-turnstore --stage base \
        --render chat --corpus gsm8k_test1319 --root data/issue_2061/hf_dl/turnstores
    uv run python scripts/issue2061_hub_io.py sentinel --kind epm:smoke-result \
        --note '{"phase": "smoke", "ok": true}'
    uv run python scripts/issue2061_hub_io.py p3-combos --r2-dir eval_results/issue_2061/per_feature_r2
    uv run python scripts/issue2061_hub_io.py --import-check
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # HF_TOKEN etc. must bind before any hub call (#397 r10')

DATA_REPO = "superkaiba1/explore-persona-space-data"

# HF data-repo bucket for this task (plan §9 `issue2061_<slug>`). Override via
# env for scratch-prefix smoke probes of the upload/stage legs (#1769
# hub-fenced-branch live-probe discipline).
HF_PREFIX = os.environ.get("ISSUE2061_HF_PREFIX", "issue2061_sae_predictability")

# Artifact class -> hub sub-prefix (plan §9 phase_outputs + off_pod_phases).
ARTIFACT_PREFIXES: dict[str, str] = {
    "sae-encoded": "sae_encoded",
    "per-feature-r2": "analysis_tensors/per_feature_r2",
    "null": "analysis_tensors/null",
    "fitness": "analysis_tensors/fitness",
}

TASK_ID = 2061


def _log(msg: str) -> None:
    """Stderr logger — stdout is reserved for machine-readable output."""
    print(msg, file=sys.stderr, flush=True)


def hub_prefix(what: str) -> str:
    """Full hub prefix for one artifact class (fail-loud on unknown class)."""
    if what not in ARTIFACT_PREFIXES:
        raise KeyError(f"unknown artifact class {what!r}; expected {sorted(ARTIFACT_PREFIXES)}")
    return f"{HF_PREFIX}/{ARTIFACT_PREFIXES[what]}"


def upload_dir(local_dir: Path | str, what: str, *, delete_local: bool = False) -> list[str]:
    """Fail-loud verified upload of every FILE in `local_dir` to the class prefix.

    Rides `upload_dir_sharded` (chunked bulk commits, exact-set verify, quota
    overflow routing — #1824/#1034). `delete_local=False` by default: the
    r2/null/fitness dirs are git-committed `eval_results/` trees and P1's
    encoded targets may still feed a same-pod P4. No eligibility filter —
    the WHOLE flat tree uploads (plan-glob parity, #825). Returns the dest
    paths uploaded or skipped-existing; raises on any verify failure.
    """
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    prefix = hub_prefix(what)
    _log(f"[upload] {local_dir} -> {DATA_REPO}/{prefix}")
    res = upload_dir_sharded(
        Path(local_dir),
        DATA_REPO,
        prefix,
        repo_type="dataset",
        shard_glob="*",
        verify=True,
        delete_local=delete_local,
    )
    dests = sorted(res.uploaded) + sorted(res.skipped_existing)
    _log(
        f"[upload] done: {len(res.uploaded)} uploaded, "
        f"{len(res.skipped_existing)} already-present, {len(res.rerouted)} rerouted-overflow"
    )
    if not dests:
        raise RuntimeError(f"upload_dir({local_dir}, {what}): no files found to upload")
    return dests


def stage_dir(what: str, staging_root: Path | str) -> Path:
    """Stage one artifact class's hub prefix; return the CONSUMED directory.

    `stage_hub_prefix` mirrors repo-relative paths under `staging_root`
    (verbatim prefix mirror), so the consumed dir is `staging_root/<prefix>`
    — returned explicitly so callers never re-derive it (#1774 mirror-root
    trap). Fail-loud on an empty prefix.
    """
    from explore_persona_space.orchestrate.hub import stage_hub_prefix

    prefix = hub_prefix(what)
    root = Path(staging_root)
    _log(f"[stage] {DATA_REPO}/{prefix} -> {root}")
    files = stage_hub_prefix(DATA_REPO, prefix, root, repo_type="dataset")
    consumed = root / prefix
    if not consumed.is_dir():
        raise FileNotFoundError(
            f"stage_dir({what}): staged {len(files)} files but consumed dir {consumed} absent "
            "— mirror-root arithmetic broken?"
        )
    _log(f"[stage] {what} staged: {len(files)} files -> {consumed}")
    return consumed


def _resolve_data_repo_revision(revision: str | None) -> str:
    """Pin ONE data-repo commit for a multi-file staging leg (#833 coherence)."""
    if revision is not None:
        return revision
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = retry_transient(
        lambda: api.repo_info(DATA_REPO, repo_type="dataset"),
        what=f"repo_info({DATA_REPO})",
    )
    return str(info.sha)


def stage_turnstore(
    stage: str,
    render: str,
    corpus: str,
    turnstore_root: Path | str,
    *,
    revision: str | None = None,
    max_shards: int | None = None,
) -> Path:
    """Stage one #1336 turnstore's `*_shardNNN.pt` files into the consumer layout.

    Shards land FLAT at `<turnstore_root>/turnstore_<stage>_<render>_<corpus>/`
    — exactly the layout `issue2061_turnstore.enumerate_shards` opens (an
    explicit hub-rel -> local-rel basename map, not a prefix mirror). One
    resolved revision covers the listing AND every file (#833). Idempotent:
    already-staged files are skipped by `stage_hub_file`. `max_shards` is a
    smoke-probe knob ONLY — a partial turnstore misaligns with a full encode
    payload, so production callers never pass it.
    """
    from explore_persona_space.orchestrate.hub import stage_hub_file

    # Sibling-script import (script-dir sys.path insert at module bottom of
    # main scripts; here the script dir IS this file's dir).
    import issue2061_sae_encode as enc

    rev = _resolve_data_repo_revision(revision)
    # Resolve the REALIZED tree name from the canonical cell identity — the
    # store carries `v2_` prefixes and the `rlvr_long` stage token, so a
    # hand-built canonical name 404s (unit-E live probe finding).
    tree_path = enc.resolve_turnstore_tree(stage, render, corpus, revision=rev)
    rels = enc.hub_shard_files(tree_path, revision=rev)
    if max_shards is not None:
        rels = rels[:max_shards]
    dest_dir = Path(turnstore_root) / f"turnstore_{stage}_{render}_{corpus}"
    t0 = time.time()
    for rel in rels:
        stage_hub_file(
            DATA_REPO,
            rel,
            dest_dir / Path(rel).name,
            repo_type="dataset",
            revision=rev,
        )
    _log(
        f"[stage] {tree_path} staged: {len(rels)} shard(s) -> {dest_dir} ({time.time() - t0:.1f}s)"
    )
    return dest_dir


def reap_turnstore(turnstore_root: Path | str, stage: str, render: str, corpus: str) -> None:
    """Delete one staged turnstore dir (stream-fetch-delete / last-consumer reap).

    Fail-loud `rmtree` (no ignore_errors — a failed reap must crash at the
    reap, #1586 fu); a missing dir logs + no-ops (already reaped / never
    staged). ONLY legal against a staging copy — callers must never point
    this at a canonical source tree.
    """
    dest_dir = Path(turnstore_root) / f"turnstore_{stage}_{render}_{corpus}"
    if not dest_dir.is_dir():
        _log(f"[reap] {dest_dir} absent — nothing to reap")
        return
    shutil.rmtree(dest_dir)
    _log(f"[reap] deleted staged {dest_dir}")


def write_sentinel(
    kind: str,
    note: str,
    *,
    sentinel_dir: Path | str | None = None,
    task_id: int = TASK_ID,
) -> Path | None:
    """Write the poll_pipeline results sentinel (pod-side reporting contract).

    Conforms to `poll_pipeline._SENTINEL_REQUIRED_KEYS`
    (`sentinel_schema_version`=1 / `kind` / `version`=1 — pod-side writers
    hardcode 1; the VM drain re-derives max+1, #1095) and lands at
    `<dir>/issue-2061-<kind_slug>-<epoch>.json`. Written ONCE per run, never
    rewritten (`.claude/rules/pod-side-reporting.md` item 3). Returns None
    (with a log line) when no sentinel dir exists — the VM-local lanes have
    no poller drain, skipping is the designed disposition there.
    """
    out_dir = Path(sentinel_dir or os.environ.get("ISSUE2061_SENTINEL_DIR", "/workspace/logs"))
    if not out_dir.is_dir():
        _log(f"[sentinel] {out_dir} absent — not a pod lane; skipping sentinel write")
        return None
    kind_slug = kind.replace(":", "_")
    path = out_dir / f"issue-{task_id}-{kind_slug}-{int(time.time())}.json"
    data = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": task_id,
        "by": "issue2061_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(data, indent=2))
    _log(f"[sentinel] wrote {path}")
    return path


def p3_combos(r2_dir: Path | str, encoded_dir: Path | str | None = None) -> list[tuple[str, str]]:
    """Realized (render, corpus) combos from the P2 R² dir, largest-first.

    Size ordering (for greedy-ish fan-out balance) uses the matching encoded
    targets' file sizes when `encoded_dir` is given — the TopK-sparse payload
    size is proportional to the cell's row count — else the R² stem order.
    """
    import issue2061_turnstore as ts

    r2 = Path(r2_dir)
    sizes: dict[tuple[str, str], int] = {}
    for path in sorted(r2.glob("*.jsonl")):
        _stage, render, corpus, _arm = ts.parse_r2_stem(path.stem, 29)
        sizes.setdefault((render, corpus), 0)
    if not sizes:
        raise FileNotFoundError(f"p3_combos: no per-feature R² files under {r2}")
    if encoded_dir is not None:
        enc_root = Path(encoded_dir)
        for render, corpus in sizes:
            total = sum(p.stat().st_size for p in enc_root.glob(f"*_{render}_{corpus}_answer_*.pt"))
            sizes[(render, corpus)] = total
    return sorted(sizes, key=lambda rc: (-sizes[rc], rc))


def _run_import_check() -> int:
    """Axis-1 import-resolution leg: execute every deferred import this module

    (and the phase scripts' upload/stage seams) hits on its REAL code path
    (`experiment-implementer.md` § smoke-architecture-check, #1689 false-pass
    class — a bare `import <module>` never fires function-body imports).
    """
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        retry_transient,
        stage_hub_file,
        stage_hub_prefix,
        verify_repo_paths_uploaded,
    )
    from explore_persona_space.orchestrate.upload_sharded import (  # noqa: F401
        upload_dir_sharded,
    )

    import issue2061_sae_encode as enc
    import issue2061_turnstore as ts

    # Signature-bind the cross-module call shapes this module composes (#1332).
    import inspect

    inspect.signature(enc.hub_shard_files).bind("tree/path", revision="r")
    inspect.signature(stage_hub_file).bind("repo", "path", Path("t"), repo_type="dataset")
    inspect.signature(stage_hub_prefix).bind("repo", "pfx", Path("root"), repo_type="dataset")
    inspect.signature(upload_dir_sharded).bind(
        Path("d"), "repo", "pfx", repo_type="dataset", verify=True, delete_local=False
    )
    inspect.signature(ts.parse_r2_stem).bind("base_chat_c_prefix_L29", 29)
    print("[import-check] OK: all deferred imports + call shapes resolve", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="#2061 cross-machine HF staging/upload wiring")
    parser.add_argument("--import-check", action="store_true", help="Axis-1 import-resolution leg")
    sub = parser.add_subparsers(dest="cmd")

    p_up = sub.add_parser("upload", help="verified upload of one artifact-class dir")
    p_up.add_argument("--what", required=True, choices=sorted(ARTIFACT_PREFIXES))
    p_up.add_argument("--dir", required=True, type=Path)
    p_up.add_argument("--delete-local", action="store_true")

    p_st = sub.add_parser("stage", help="stage one artifact class; prints the consumed dir")
    p_st.add_argument("--what", required=True, choices=sorted(ARTIFACT_PREFIXES))
    p_st.add_argument("--root", required=True, type=Path)

    p_ts = sub.add_parser("stage-turnstore", help="stage one #1336 turnstore into consumer layout")
    p_ts.add_argument("--stage", required=True)
    p_ts.add_argument("--render", required=True)
    p_ts.add_argument("--corpus", required=True)
    p_ts.add_argument("--root", required=True, type=Path)
    p_ts.add_argument("--data-revision", default=None)
    p_ts.add_argument("--max-shards", type=int, default=None, help="SMOKE-PROBE ONLY")

    p_se = sub.add_parser("sentinel", help="write the poll_pipeline end-of-run sentinel")
    p_se.add_argument("--kind", required=True)
    p_se.add_argument("--note", default=None)
    p_se.add_argument("--note-file", type=Path, default=None)
    p_se.add_argument("--dir", type=Path, default=None)

    p_c = sub.add_parser("p3-combos", help="print realized 'render corpus' combos, largest-first")
    p_c.add_argument("--r2-dir", required=True, type=Path)
    p_c.add_argument("--encoded-dir", type=Path, default=None)

    args = parser.parse_args()
    if args.import_check:
        return _run_import_check()
    if args.cmd == "upload":
        upload_dir(args.dir, args.what, delete_local=args.delete_local)
        return 0
    if args.cmd == "stage":
        print(stage_dir(args.what, args.root))
        return 0
    if args.cmd == "stage-turnstore":
        dest = stage_turnstore(
            args.stage,
            args.render,
            args.corpus,
            args.root,
            revision=args.data_revision,
            max_shards=args.max_shards,
        )
        print(dest)
        return 0
    if args.cmd == "sentinel":
        if (args.note is None) == (args.note_file is None):
            parser.error("sentinel: pass exactly one of --note / --note-file")
        note = args.note if args.note is not None else args.note_file.read_text()
        write_sentinel(args.kind, note, sentinel_dir=args.dir)
        return 0
    if args.cmd == "p3-combos":
        for render, corpus in p3_combos(args.r2_dir, args.encoded_dir):
            print(f"{render} {corpus}")
        return 0
    parser.error("pass a subcommand or --import-check")
    return 2


# Sibling-script imports (issue2061_sae_encode / issue2061_turnstore) resolve
# via the script-dir path insert — same pattern as the other #2061 scripts.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


if __name__ == "__main__":
    sys.exit(main())
