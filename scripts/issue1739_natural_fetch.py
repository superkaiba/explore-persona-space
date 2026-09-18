"""Fetch compact completed results and independently verify immutable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"


def fetch(observation_path: Path, destination: Path):
    observed = json.loads(observation_path.read_text())
    reports = {}
    for behavior, state in observed["backend_observation"]["handles"].items():
        if state["status"] != "complete":
            continue
        identity = state["results"]
        prefix, revision = identity["prefix"], identity["verified_revision"]
        receipt_path = hub.retry_transient(
            lambda: hf_hub_download(
                REPO,
                prefix + ".completion.json",
                repo_type="dataset",
                revision=identity["receipt_publication_revision"],
            ),
            what=f"natural completion receipt: {behavior}",
        )
        receipt = json.loads(Path(receipt_path).read_text())
        for key in ("source_sha", "input_fingerprint", "verified_revision"):
            if receipt[key] != identity[key]:
                raise ValueError(f"Receipt identity mismatch: {behavior}/{key}")
        expected = receipt["verification"]["sha256"]
        entries = {
            e.path[len(prefix) + 1 :]: e
            for e in hub.retry_transient(
                lambda: list(
                    HfApi().list_repo_tree(
                        REPO,
                        path_in_repo=prefix,
                        repo_type="dataset",
                        revision=revision,
                        recursive=True,
                    )
                ),
                what=f"natural output inventory: {behavior}",
            )
            if hasattr(e, "size")
        }
        if set(entries) != set(expected):
            raise ValueError(f"Immutable output name-set mismatch: {behavior}")
        target = destination / behavior
        target.mkdir(parents=True, exist_ok=True)

        def one(item):
            name, entry = item
            retained = name.endswith(".json") or Path(name).name.startswith(
                ("predictions_", "bootstrap_")
            )
            if entry.lfs is not None and entry.lfs.sha256 != expected[name]:
                raise ValueError(f"Remote tensor digest mismatch: {behavior}/{name}")
            if retained or entry.lfs is None:
                downloaded = Path(
                    hub.retry_transient(
                        lambda: hf_hub_download(
                            REPO, prefix + "/" + name, repo_type="dataset", revision=revision
                        ),
                        what=f"natural output file: {behavior}/{name}",
                    )
                )
                with downloaded.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != expected[name] or downloaded.stat().st_size != entry.size:
                    raise ValueError(f"Downloaded digest/size mismatch: {behavior}/{name}")
                if retained:
                    path = target / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(downloaded.read_bytes())
            return name, {"sha256": expected[name], "bytes": entry.size, "retained": retained}

        with ThreadPoolExecutor(max_workers=6) as pool:
            files = dict(pool.map(one, entries.items()))
        report = {"repo": REPO, **identity, "files": files}
        (target / "fetch_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        (target / "published_completion.json").write_bytes(Path(receipt_path).read_bytes())
        reports[behavior] = {
            "revision": revision,
            "verified_files": len(files),
            "verified_bytes": sum(f["bytes"] for f in files.values()),
            "retained_bytes": sum(f["bytes"] for f in files.values() if f["retained"]),
        }
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    fetch(args.observation, args.destination)
