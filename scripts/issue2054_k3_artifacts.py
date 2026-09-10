"""Verified batch commits for K3 checkpoints, reducing Hub commit pressure."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hashlib
from pathlib import Path

from scripts import issue2054_k3 as k3


def seal_many(paths, root, fingerprint):
    from huggingface_hub import CommitOperationAdd, HfApi
    from explore_persona_space.orchestrate.hub import retry_transient

    paths = [Path(p) for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("checkpoint packet must be nonempty and unique")
    api = HfApi()

    def commit(pairs):
        # Rebuild operations on retries: the Hub mutates their upload state.
        revision = retry_transient(
            lambda: api.create_commit(
                repo_id=k3.HF_REPO,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in pairs
                ],
                commit_message=f"#2054 verified checkpoint packet ({len(pairs)} files)",
            ),
            what="K3 checkpoint packet commit",
        ).oid
        entries = retry_transient(
            lambda: api.get_paths_info(
                k3.HF_REPO, [dst for _, dst in pairs], repo_type="dataset", revision=revision
            ),
            what="K3 checkpoint packet verify",
        )
        by_path = {e.path: e for e in entries}
        for src, dst in pairs:
            entry = by_path[dst]
            if entry.size != src.stat().st_size:
                raise RuntimeError(f"uploaded checkpoint size mismatch: {dst}")
            lfs_hash = getattr(getattr(entry, "lfs", None), "sha256", None)
            if lfs_hash is not None:
                correct = lfs_hash == k3.sha(src)
            else:
                data = src.read_bytes()
                correct = (
                    entry.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                )
            if not correct:
                raise RuntimeError(f"uploaded checkpoint hash mismatch: {dst}")
        return revision

    pairs = [(p, f"{k3.PREFIX}/{root.name}/{p.relative_to(root)}") for p in paths]
    revision = commit(pairs)
    pending = []
    for path, destination in pairs:
        done = path.with_suffix(path.suffix + ".done.json")
        temp = done.with_suffix(".pending")
        k3.atomic_json(
            temp,
            {
                "path": destination,
                "revision": revision,
                "sha256": k3.sha(path),
                "size": path.stat().st_size,
                "fingerprint": fingerprint,
            },
        )
        pending.append((temp, destination + ".done.json", done))
    commit([(temp, dst) for temp, dst, _ in pending])
    for temp, _, done in pending:
        temp.replace(done)
    k3.log(f"[phase=checkpoint_packet] verified files={len(paths)} revision={revision}")
