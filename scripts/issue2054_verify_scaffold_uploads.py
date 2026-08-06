#!/usr/bin/env python
"""CONTENT-aware upload verification for #2054's phase_a scaffold artifacts.

Why this exists (r6 incident, 2026-08-05): the round-5 upload verification was
presence-only against path names composed from the PRE-shard convention
(`scaffolds_<v>_prejudge.jsonl`). Those names do exist on HF — as the previous
round's smaller, UNSHARDED residue — so the check matched stale files, returned
PASS, and green-lit a pod teardown. It was correct only by luck: had the gen leg
genuinely failed to upload, that PASS would have authorised destroying the only
copy.

`upload-policy.md` is explicit that a post-upload verify is an EXACT
expected-file-set check on a fresh listing, "NOT prefix-presence / count-only".
This script is that check for the shapes phase_a actually writes:

  - text >9.5 MB is line-split by `_shard_large_jsonl_for_upload` into
    `<stem>.shardNN.jsonl` + `<stem>.manifest.json`, so the expected set is
    DERIVED FROM THE MANIFEST, never from a name convention;
  - every shard's sha256 is verified against the manifest;
  - the reassembled row count is reconciled against an EXPECTED count
    (the gen leg's own `gen digest`), so a complete-but-WRONG upload fails.

Exit 0 = PASS, 1 = FAIL (with every miss named), 2 = usage/transport error.

Usage:
  uv run python scripts/issue2054_verify_scaffold_uploads.py \
      --expect char_helios=6931 --expect char_wren=6916 ...
  (omit --expect to verify structure + shard shas only, and say so in the
   verdict — a structure-only pass is NOT a content guarantee)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2054_lattice/scaffolds"
DEFAULT_VARIANTS = (
    "char_helios",
    "char_wren",
    "char_dana",
    "char_vex",
    "conversation_paired_stories_assistant",
)


def _log(msg: str) -> None:
    print(f"[verify-uploads] {msg}", flush=True)


def _stage(api, path_in_repo: str, dest: Path):
    from explore_persona_space.orchestrate.hub import stage_hub_file

    return stage_hub_file(HF_DATA_REPO, path_in_repo, dest, repo_type="dataset", overwrite=True)


def _verify_manifest_shards(
    api, listing: set[str], work_dir: Path, base: str, stem: str, label: str
):
    """Manifest-first shard verification (shared by the variant pools and the
    shared question draw): every part the manifest names must be on HF and
    sha256-match the manifest. Returns (problems, n_rows_or_None)."""
    problems: list[str] = []
    n_rows: int | None = None
    mlocal = _stage(api, f"{base}/{stem}.manifest.json", work_dir / f"{stem}.manifest.json")
    man = json.loads(Path(mlocal).read_text(encoding="utf-8"))
    parts = list(man.get("parts") or [])
    shas = man.get("sha256") or {}
    counts = list(man.get("line_counts") or [])
    if not parts:
        problems.append(f"{label}: manifest lists no parts")
    for name in parts:
        remote = f"{base}/{name}"
        if remote not in listing:
            problems.append(f"{label}: manifest names {name} but it is NOT on HF")
            continue
        local = Path(_stage(api, remote, work_dir / name))
        got = hashlib.sha256(local.read_bytes()).hexdigest()
        want = shas.get(name)
        if want and got != want:
            problems.append(f"{label}: shard {name} sha {got[:12]} != manifest {str(want)[:12]}")
    if counts and len(counts) == len(parts):
        n_rows = sum(int(c) for c in counts)
    return problems, n_rows


def verify_variant(api, variant: str, listing: set[str], work: Path, expect: int | None):
    """Return (ok, [problems], n_rows_or_None) for one variant."""
    problems: list[str] = []
    stem = f"scaffolds_{variant}_prejudge"
    base = f"{PREFIX}/{variant}"

    sidecar = f"{base}/{stem}.jsonl.done.json"
    if sidecar not in listing:
        problems.append(f"missing staleness sidecar: {sidecar}")

    manifest_path = f"{base}/{stem}.manifest.json"
    sharded = manifest_path in listing
    n_rows: int | None = None

    if sharded:
        probs, n_rows = _verify_manifest_shards(api, listing, work / variant, base, stem, variant)
        problems.extend(probs)
    else:
        plain = f"{base}/{stem}.jsonl"
        if plain not in listing:
            problems.append(f"{variant}: neither a manifest NOR {stem}.jsonl on HF")
        else:
            local = Path(_stage(api, plain, work / variant / f"{stem}.jsonl"))
            n_rows = sum(
                1 for line in local.read_text(encoding="utf-8").split("\n") if line.strip()
            )
            problems.append(
                f"{variant}: NOTE unsharded form — expected sharded above 9.5 MB; "
                "confirm this pool is genuinely small"
            )

    if expect is not None:
        if n_rows is None:
            problems.append(f"{variant}: cannot determine row count to reconcile against {expect}")
        elif n_rows != expect:
            problems.append(f"{variant}: rows {n_rows} != expected {expect} (STALE or partial)")

    return (not problems), problems, n_rows


def verify_shared_draw(api, listing: set[str], work: Path, expect: int | None):
    """Return (ok, [problems], n_rows_or_None) for the seed-137 shared draw.

    MANIFEST-FIRST (the r15 fix, closing the draw-path r6 blind spot): the
    top-up gen leg uploads the >9.5 MB draw in the SHARDED form only, so when
    a ``shared_question_draw.manifest.json`` exists the plain hub name is
    stale prior-round residue and its presence proves NOTHING — a
    presence-only check on it green-lit exactly the staleness this script
    exists to close (epm:failure v4). Unsharded presence is accepted only
    when no manifest exists (pre-shard compat), with the same
    confirm-it-is-genuinely-small NOTE as the variant pools."""
    problems: list[str] = []
    stem = "shared_question_draw"
    if f"{PREFIX}/{stem}.meta.json" not in listing:
        problems.append(f"missing shared draw artifact: {stem}.meta.json")
    n_rows: int | None = None
    if f"{PREFIX}/{stem}.manifest.json" in listing:
        probs, n_rows = _verify_manifest_shards(
            api, listing, work / "shared_draw", PREFIX, stem, "shared_draw"
        )
        problems.extend(probs)
    elif f"{PREFIX}/{stem}.jsonl" in listing:
        local = Path(_stage(api, f"{PREFIX}/{stem}.jsonl", work / "shared_draw" / f"{stem}.jsonl"))
        n_rows = sum(1 for line in local.read_text(encoding="utf-8").split("\n") if line.strip())
        problems.append(
            "shared_draw: NOTE unsharded form only — expected sharded above 9.5 MB; "
            "confirm this draw is genuinely small"
        )
    else:
        problems.append(f"shared_draw: neither a manifest NOR {stem}.jsonl on HF")
    if expect is not None:
        if n_rows is None:
            problems.append(
                f"shared_draw: cannot determine row count to reconcile against {expect}"
            )
        elif n_rows != expect:
            problems.append(f"shared_draw: rows {n_rows} != expected {expect} (STALE or partial)")
    return (not problems), problems, n_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="VARIANT=N",
        help="expected row count per variant (from the gen leg's digest line)",
    )
    ap.add_argument(
        "--expect-draw",
        type=int,
        default=None,
        metavar="N",
        help="expected row count for the shared question draw (from the gen digest)",
    )
    ap.add_argument("--work-dir", default="/tmp/issue-2054-verify-uploads")
    args = ap.parse_args()

    expect: dict[str, int] = {}
    for spec in args.expect:
        if "=" not in spec:
            print(f"ERROR: --expect wants VARIANT=N, got {spec!r}", file=sys.stderr)
            return 2
        k, v = spec.split("=", 1)
        expect[k.strip()] = int(v)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    api = HfApi()
    listing = set(list_hf_files_under_path(api, HF_DATA_REPO, PREFIX, repo_type="dataset"))
    _log(f"scoped listing: {len(listing)} files under {PREFIX}")

    all_problems: list[str] = []
    for v in variants:
        ok, problems, n_rows = verify_variant(api, v, listing, work, expect.get(v))
        rows = "?" if n_rows is None else str(n_rows)
        _log(f"{v:40s} rows={rows:>6s} {'OK' if ok else 'PROBLEMS'}")
        all_problems.extend(problems)

    ok, problems, n_rows = verify_shared_draw(api, listing, work, args.expect_draw)
    rows = "?" if n_rows is None else str(n_rows)
    _log(f"{'shared_question_draw':40s} rows={rows:>6s} {'OK' if ok else 'PROBLEMS'}")
    all_problems.extend(problems)

    if all_problems:
        _log("VERDICT: FAIL")
        for p in all_problems:
            _log(f"  - {p}")
        return 1

    if not expect and args.expect_draw is None:
        _log(
            "VERDICT: PASS (STRUCTURE ONLY — no --expect row counts supplied, so this "
            "does NOT prove the upload is this round's content)"
        )
    else:
        _log("VERDICT: PASS (structure + shard sha256 + row counts reconciled)")
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(main())
