#!/usr/bin/env python
"""#825 turn-dynamics upload helper (one bulk upload_folder commit per call).

Env contract (set by the dispatcher queue): SRC (local dir), DEST (repo
prefix), SMOKE (any non-empty value skips the network). Modes:

  --mode text     JSONL/JSON payloads. Any *.jsonl > 9.5 MB is line-split into
                  < 9 MB `.part{NN}.jsonl` shards BEFORE upload (non-LFS path;
                  never gzip — upload-policy.md), then ONE upload_folder
                  commit + a scoped fresh-listing verify.
  --mode tensors  fp16 npy stores / betas / reports: ONE upload_folder commit
                  (LFS path) + scoped verify.

Bounded outer backoff (5 attempts, 60s*2^k + jitter) around the commit; the
verify listing uses the server-side scoped list_repo_tree (never a bare
list_repo_files on the ~1M-file repo — gotchas.md).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

SHARD_MAX_BYTES = 9_000_000
SPLIT_THRESHOLD = 9_500_000


def _split_oversize_jsonl(src: Path) -> list[Path]:
    """Line-split every > 9.5 MB *.jsonl under src into < 9 MB parts (in place).

    The original file is renamed to `<name>.orig.local` (excluded from upload)
    so re-runs are idempotent. Returns the list of part files written.
    """
    written: list[Path] = []
    for p in sorted(src.rglob("*.jsonl")):
        if p.stat().st_size <= SPLIT_THRESHOLD or ".part" in p.name:
            continue
        parts: list[Path] = []
        si, size, f = 0, 0, None
        try:
            with p.open(encoding="utf-8") as fin:
                for line in fin:
                    raw = line if line.endswith("\n") else line + "\n"
                    if f is None or size + len(raw.encode()) > SHARD_MAX_BYTES:
                        if f is not None:
                            f.close()
                        part = p.with_name(f"{p.stem}.part{si:02d}.jsonl")
                        f = part.open("w", encoding="utf-8")
                        parts.append(part)
                        si, size = si + 1, 0
                    f.write(raw)
                    size += len(raw.encode())
        finally:
            if f is not None:
                f.close()
        manifest = {
            "original": p.name,
            "parts": [q.name for q in parts],
            "n_parts": len(parts),
        }
        p.with_name(f"{p.stem}.manifest.json").write_text(json.dumps(manifest, indent=1))
        p.rename(p.with_name(p.name + ".orig.local"))
        written.extend(parts)
        print(f"[upload] split {p.name} -> {len(parts)} parts")
    return written


def _with_backoff(label: str, fn) -> None:
    last: Exception | None = None
    for attempt in range(5):
        try:
            fn()
            return
        except Exception as e:
            last = e
            wait = 60 * (2**attempt) + random.uniform(0, 15)
            print(
                f"[upload] {label} attempt {attempt + 1}/5 failed "
                f"({type(e).__name__}); retrying in {wait:.0f}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"[upload] {label} FAILED after 5 attempts") from last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("text", "tensors"))
    args = ap.parse_args()
    src = Path(os.environ["SRC"])
    dest = os.environ["DEST"]
    if os.environ.get("SMOKE"):
        print(f"[upload] smoke: skipping HF upload of {src} -> {dest}")
        return
    repo = "superkaiba1/explore-persona-space-data"
    from huggingface_hub import HfApi, upload_folder

    if args.mode == "text":
        _split_oversize_jsonl(src)
        allow = ["**/*.jsonl", "**/*.json", "*.jsonl", "*.json"]
        ignore = ["**/*.orig.local", "**/haiku_cache/**", "**/haiku_ckpt/**"]
    else:
        allow = ["**/*.npy", "**/*.json", "*.npy", "*.json"]
        ignore = ["**/*.orig.local"]

    _with_backoff(
        f"{args.mode}:{dest}",
        lambda: upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(src),
            path_in_repo=dest,
            allow_patterns=allow,
            ignore_patterns=ignore,
            commit_message=f"issue-825 turn-dynamics: {args.mode} -> {dest}",
        ),
    )
    # scoped fresh-listing verify (exact filename containment)
    api = HfApi()
    local = sorted(
        str(p.relative_to(src))
        for p in src.rglob("*")
        if p.is_file()
        and (
            (args.mode == "text" and p.suffix in (".jsonl", ".json"))
            or (args.mode == "tensors" and p.suffix in (".npy", ".json"))
        )
        and ".orig.local" not in p.name
        and "haiku_cache" not in str(p)
        and "haiku_ckpt" not in str(p)
    )
    hub = {
        e.path.removeprefix(dest + "/")
        for e in api.list_repo_tree(repo, path_in_repo=dest, repo_type="dataset", recursive=True)
    }
    missing = [f for f in local if f not in hub]
    assert not missing, f"[upload] files missing on Hub under {dest}: {missing[:10]}"
    print(f"[upload] verified {len(local)} files @ {dest}")


if __name__ == "__main__":
    main()
