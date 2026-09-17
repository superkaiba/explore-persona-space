#!/usr/bin/env python3
"""Render, persist, and verify the task 2673 context-geometry pilot artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.hf_api import RepoFile  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.backends.artifacts import write_completion_sentinel  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from scripts.story_persona_qwen38_pilot import (  # noqa: E402
    digest,
    file_digest,
    read_manifest,
    write_json,
)

ISSUE = 2673
PREFIX = "issue2673_story_persona_qwen38/analysis_tensors"
BRANCH = "codex/story-persona-qwen38-pilot-20260917"
GITHUB = "https://github.com/superkaiba/explore-persona-space"
FIXED_LAYERS = [15, 31, 47, 63]
STEMS = ["centered_cosine_fixed_layers", "sfl_similarity_by_layer"]
WANDB_ALIASES = ("wandb/latest-run", "wandb/debug.log", "wandb/debug-internal.log")
LABELS = {
    "default": "Default",
    "sarcasm": "Sarcasm",
    "sarcasm_lists": "Sarcasm + lists",
    "sfl": "Full SFL",
    "french": "French",
    "french_lists": "French + lists",
    "persona_dismissive": "Dismissive",
    "persona_sarcastic": "Sarcastic",
    "persona_saboteur": "Saboteur",
    "persona_peer": "Peer",
}


def inventory(root: Path) -> dict[str, dict]:
    """Inventory all regular files, rejecting symlinks and unstable write remnants."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            relative = path.relative_to(root).as_posix()
            target = path.resolve(strict=True)
            if relative in WANDB_ALIASES and target.is_relative_to((root / "wandb").resolve()):
                continue  # Actual target is inventoried; record convenience alias separately.
            raise RuntimeError(f"refusing symlink in upload tree: {path}")
        if not path.is_file():
            continue
        if path.suffix == ".tmp" or ".tmp." in path.name:
            raise RuntimeError(f"unfinished output: {path}")
        size = path.stat().st_size
        sha1 = hashlib.sha1(f"blob {size}\0".encode())
        sha256 = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha1.update(block)
                sha256.update(block)
        result[path.relative_to(root).as_posix()] = {
            "size": size,
            "sha256": sha256.hexdigest(),
            "git_blob_sha1": sha1.hexdigest(),
        }
    if not result:
        raise RuntimeError("empty artifact tree")
    return result


def validate_inputs(out: Path) -> tuple[dict, dict, dict]:
    """Bind plotted values and every persisted chunk to the completed capture recipe."""
    manifest = read_manifest(out / "manifest.json")
    spec = manifest["spec"]
    summary = json.loads((out / "summary.json").read_text())
    done = json.loads((out / "capture_complete.json").read_text())
    rows = json.loads((out / "rows.json").read_text())
    if any(x["fingerprint"] != manifest["fingerprint"] for x in (summary, done)):
        raise RuntimeError("mixed capture/analysis fingerprints")
    names = [p["id"] for p in spec["prompts"]]
    expected_rows = {(p, q) for p in names for q in spec["question_ids"]}
    actual_rows = [(r["persona"], r["question_id"]) for r in rows]
    if (
        digest(rows) != spec["inputs_sha256"]
        or set(actual_rows) != expected_rows
        or len(actual_rows) != len(expected_rows)
        or done["row_count"] != len(rows)
    ):
        raise RuntimeError("row identity or coverage mismatch")
    if summary["persona_names"] != names or summary["layers"] != list(range(64)):
        raise RuntimeError("unexpected persona order or layer coverage")
    if names != list(LABELS) or summary["question_count"] != len(spec["question_ids"]):
        raise RuntimeError("unexpected pilot persona bank or question count")
    files = inventory(out)
    chunks = {f"batch_{i:04d}.pt" for i in range(len(spec["batches"]))}
    if (
        set(done["chunk_sha256"]) != chunks
        or {p.removeprefix("chunks/") for p in files if p.startswith("chunks/")} != chunks
    ):
        raise RuntimeError("chunk filename coverage mismatch")
    for name, expected in done["chunk_sha256"].items():
        if files[f"chunks/{name}"]["sha256"] != expected:
            raise RuntimeError(f"corrupt completed chunk: {name}")
    with np.load(out / "centroids.npz", allow_pickle=False) as data:
        centers = data["centroids"]
        if centers.shape != (10, 64, spec["model"]["hidden_dim"]):
            raise RuntimeError("centroid shape mismatch")
        centered = centers - centers.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=-1, keepdims=True)
        if not np.isfinite(centered).all() or np.any(norms == 0):
            raise RuntimeError("degenerate or non-finite centroids")
        centered /= norms
        expected_cosine = np.einsum("plh,qlh->lpq", centered, centered)
        actual_cosine = np.asarray(summary["centered_cosine"])
        if actual_cosine.shape != (64, 10, 10) or not np.allclose(
            actual_cosine, expected_cosine, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("summary cosine differs from persisted centroids")
    return manifest, summary, files


def save_plot(fig, stem: Path, title: str, manifest: dict, summary: dict, out: Path) -> None:
    """Export the established figure formats with exact input/output provenance."""
    saved = save_c2a_figure(
        fig,
        stem,
        title=title,
        subject="Persona-prompted context geometry; no behavioral leakage measurement",
        creator="scripts/story_persona_qwen38_artifacts.py",
    )
    write_json(
        stem.with_suffix(".meta.json"),
        {
            "fingerprint": manifest["fingerprint"],
            "source_sha": manifest["provenance"]["git_commit"],
            "summary_sha256": file_digest(out / "summary.json"),
            "centroids_sha256": file_digest(out / "centroids.npz"),
            "script_sha256": file_digest(Path(__file__)),
            "fixed_layers_zero_based": FIXED_LAYERS,
            "centering": "global mean across all ten persona centroids, separately per layer",
            "question_count": summary["question_count"],
            "interpretation": summary["interpretation"],
            "render": saved["record"],
            "output_sha256": {k: file_digest(Path(saved[k])) for k in ("png", "pdf", "grayscale")},
        },
    )
    plt.close(fig)


def prepare(out: Path, repo: Path) -> dict:
    """Render fixed, prespecified summaries and a complete local artifact manifest."""
    manifest, summary, files = validate_inputs(out)
    figures = repo / f"figures/issue_{ISSUE}"
    results = repo / f"eval_results/issue_{ISSUE}"
    names = summary["persona_names"]
    cosine = np.asarray(summary["centered_cosine"])
    set_c2a_style()
    fig, _ = c2a_figure("full", aspect=1.30)
    axes = fig.subplots(2, 2)
    for ax, layer, letter in zip(axes.flat, FIXED_LAYERS, "ABCD", strict=True):
        im = ax.imshow(cosine[layer], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(10), [LABELS[n] for n in names], rotation=65, ha="right")
        ax.set_yticks(range(10), [LABELS[n] for n in names])
        panel_header(ax, letter, f"Block {layer} (zero-based)", kicker_y=1.04)
    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.19, top=0.96, hspace=0.72, wspace=0.69)
    cax = fig.add_axes((0.27, 0.035, 0.55, 0.015))
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Centered cosine similarity")
    save_plot(
        fig, figures / STEMS[0], "Persona context cosine at fixed layers", manifest, summary, out
    )

    fig, _ = c2a_figure("full", aspect=0.52)
    ax = fig.subplots()
    target = names.index("sfl")
    primary = [p["id"] for p in manifest["spec"]["prompts"] if p["primary"]]
    for i, (name, marker) in enumerate(zip(primary, ("o", "s", "^", "D", "v", "P"), strict=True)):
        ax.plot(
            summary["layers"],
            cosine[:, names.index(name), target],
            label=LABELS[name],
            marker=marker,
            markevery=8,
            linewidth=2,
            linestyle=("-", "--", "-.")[i % 3],
        )
    style_axis(ax)
    ax.set(
        xlabel="Decoder block (zero-based)",
        ylabel="Centered cosine with full SFL",
        ylim=(-1.05, 1.05),
    )
    panel_header(
        ax, "", "Ten-persona centering", "Similarity to sarcasm + French + lists", kicker_y=1.18
    )
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.30, top=0.80)
    save_plot(
        fig,
        figures / STEMS[1],
        "Similarity to the full SFL persona across layers",
        manifest,
        summary,
        out,
    )
    write_json(results / "summary.json", summary)
    payload = {
        "issue": ISSUE,
        "fingerprint": manifest["fingerprint"],
        "source_sha": manifest["provenance"]["git_commit"],
        "hf_repo": hub.DEFAULT_DATASET_REPO,
        "hf_prefix": PREFIX,
        "files": files,
        "excluded_convenience_aliases": {
            alias: (out / alias).resolve().relative_to(out.resolve()).as_posix()
            for alias in WANDB_ALIASES
            if (out / alias).is_symlink()
        },
        "row_count": len(names) * summary["question_count"],
        "fixed_layers_zero_based": FIXED_LAYERS,
        "interpretation": summary["interpretation"],
    }
    write_json(results / "artifact_manifest.json", payload)
    return payload


def verify_hub_entries(entries: list, expected: dict, prefix: str) -> None:
    """Verify exact names, byte counts, and LFS SHA256 or ordinary Git blob hashes."""
    remote = {entry.path: entry for entry in entries if isinstance(entry, RepoFile)}
    wanted = {f"{prefix}/{name}": record for name, record in expected.items()}
    if set(remote) != set(wanted):
        raise RuntimeError(
            f"Hub file-set mismatch: missing={set(wanted) - set(remote)}, extra={set(remote) - set(wanted)}"
        )
    for name, record in wanted.items():
        entry = remote[name]
        if entry.size != record["size"]:
            raise RuntimeError(f"Hub size mismatch: {name}")
        lfs = entry.lfs
        if lfs is not None:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            expected_hash = record["sha256"]
        else:
            sha, expected_hash = entry.blob_id, record["git_blob_sha1"]
        if sha != expected_hash:
            raise RuntimeError(f"Hub content hash mismatch: {name}")


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Execute bounded Git operations without shell interpolation or credential output."""
    proc = subprocess.run(
        ["git", "--literal-pathspecs", *args], cwd=repo, capture_output=True, text=True, timeout=180
    )
    if check and proc.returncode:
        raise RuntimeError(f"git {args[0]} failed with exit {proc.returncode}")
    return proc


def push_results(repo: Path, paths: list[Path]) -> str:
    """Commit explicit files, push with bounded rebase retries, then check remote blobs."""
    branch = git(repo, "branch", "--show-current").stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"results must stay on {BRANCH}; current branch is {branch!r}")
    rels = [p.resolve().relative_to(repo.resolve()).as_posix() for p in paths]
    local_blobs = {rel: git(repo, "hash-object", "--", rel).stdout.strip() for rel in rels}
    git(repo, "add", "--", *rels)
    changed = git(repo, "diff", "--cached", "--quiet", "--", *rels, check=False)
    if changed.returncode == 1:
        git(
            repo,
            "commit",
            "-m",
            f"task #{ISSUE}: verified persona context pilot artifacts",
            "--",
            *rels,
        )
    elif changed.returncode != 0:
        raise RuntimeError("cannot inspect staged result files")
    for attempt in range(2):
        git(repo, "fetch", "origin", branch)
        rebased = git(repo, "rebase", f"origin/{branch}", check=False)
        if rebased.returncode:
            git(repo, "rebase", "--abort")
            raise RuntimeError("result branch rebase conflicted")
        if git(repo, "push", "origin", f"HEAD:{branch}", check=False).returncode == 0:
            break
        if attempt == 1:
            raise RuntimeError("result push failed after two attempts")
    git(repo, "fetch", "origin", branch)
    revision = git(repo, "rev-parse", f"origin/{branch}").stdout.strip()
    if git(repo, "rev-list", "--count", f"{revision}..HEAD").stdout.strip() != "0":
        raise RuntimeError("result commits remain unpushed")
    for rel, expected in local_blobs.items():
        if git(repo, "rev-parse", f"{revision}:{rel}").stdout.strip() != expected:
            raise RuntimeError(f"remote Git content differs: {rel}")
    return revision


def publish(out: Path, repo: Path, logs: Path, sentinel_path: Path) -> None:
    """Persist every artifact and emit completion only after immutable content verification."""
    artifacts = prepare(out, repo)
    source_sha = os.environ["EPS_STORY_PERSONA_SOURCE_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or source_sha != artifacts["source_sha"]:
        raise RuntimeError("capture provenance does not match pinned launch source")
    expected = artifacts["files"]
    expected_paths = [f"{PREFIX}/{name}" for name in expected]
    print("[phase=upload] bulk artifact upload", flush=True)
    destination = hub._upload_folder_filtered(
        out,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        PREFIX,
        allow_patterns=["*"],
        ignore_patterns=[*WANDB_ALIASES, "wandb/latest-run/**"],
        expected_repo_paths=expected_paths,
        delete_after=False,
    )
    if not destination or not destination.endswith("/" + PREFIX):
        raise RuntimeError("bulk upload failed or returned an unexpected destination")
    actual_repo = destination[: -len("/" + PREFIX)]
    api = HfApi()
    revision = hub._retry_upload(
        lambda: api.repo_info(actual_repo, repo_type="dataset").sha,
        what="resolve uploaded revision",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Hub upload has no immutable revision")
    entries = hub._retry_upload(
        lambda: list(
            api.list_repo_tree(
                actual_repo,
                repo_type="dataset",
                revision=revision,
                path_in_repo=PREFIX,
                recursive=True,
            )
        ),
        what="verify uploaded artifact hashes",
    )
    verify_hub_entries(entries, expected, PREFIX)
    if inventory(out) != expected:
        raise RuntimeError("local artifact tree changed during upload")
    results = repo / f"eval_results/issue_{ISSUE}"
    figures = repo / f"figures/issue_{ISSUE}"
    artifacts["canonical_hf_repo"] = hub.DEFAULT_DATASET_REPO
    artifacts["hf_repo"] = actual_repo
    artifacts["verified_revision"] = revision
    write_json(results / "artifact_manifest.json", artifacts)
    receipt = {
        "issue": ISSUE,
        "fingerprint": artifacts["fingerprint"],
        "source_sha": source_sha,
        "verified_revision": revision,
        "hf_repo": actual_repo,
        "hf_prefix": PREFIX,
        "repo_id": actual_repo,
        "prefix": PREFIX,
        "file_count": len(expected),
        "total_bytes": sum(r["size"] for r in expected.values()),
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verification": "Exact file set and size; LFS SHA256 or non-LFS Git blob SHA1; local SHA256 manifest",
        "url": f"https://huggingface.co/datasets/{actual_repo}/tree/{revision}/{PREFIX}",
    }
    write_json(results / "upload_receipt.json", receipt)
    paths = [
        results / name for name in ("summary.json", "artifact_manifest.json", "upload_receipt.json")
    ]
    paths += [
        figures / (stem + suffix)
        for stem in STEMS
        for suffix in (".png", ".pdf", "_grayscale.png", ".meta.json")
    ]
    print("[phase=results_push] verify remote result bytes", flush=True)
    git_revision = push_results(repo, paths)
    completion = {
        **receipt,
        "phase": "done",
        "result_git_revision": git_revision,
        "git_paths": [p.relative_to(repo).as_posix() for p in paths],
        "row_count": artifacts["row_count"],
        "figure_urls": [
            f"{GITHUB}/blob/{git_revision}/figures/issue_{ISSUE}/{stem}.png" for stem in STEMS
        ],
        "interpretation": artifacts["interpretation"],
    }
    # This durable Git receipt attests only already verified revisions. The
    # backend sentinel is emitted after the receipt itself is pushed and checked.
    write_json(results / "completion_receipt.json", completion)
    final_git_revision = push_results(repo, [results / "completion_receipt.json"])
    completion["completion_receipt_git_revision"] = final_git_revision
    temporary = sentinel_path.with_suffix(".tmp")
    write_completion_sentinel(sentinel_path=temporary, issue=ISSUE, extra=completion)
    temporary.replace(sentinel_path)
    logs.mkdir(parents=True, exist_ok=True)
    sentinel = logs / f"issue-{ISSUE}-epm_results-{time.time_ns()}.json"
    write_json(
        sentinel,
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "issue": ISSUE,
            "task_id": ISSUE,
            "blocks_pipeline": False,
            "gate": "results",
            "note": completion,
        },
    )
    print(f"[artifacts] verified HF {revision} and Git {final_git_revision}", flush=True)


def main() -> None:
    """Run offline rendering or the approved remote upload/report phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "publish"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--logs-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--sentinel-path", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.out_dir.resolve(), args.repo_root.resolve())
        return
    sentinel = args.sentinel_path
    declared_path = os.environ.get("EPS_SENTINEL_PATH")
    if declared_path:
        declared = Path(declared_path)
        if sentinel is not None and sentinel != declared:
            raise RuntimeError("explicit and dispatch-declared sentinel paths differ")
        sentinel = declared
    if sentinel is None:
        raise RuntimeError("publish requires the dispatch-declared completion sentinel path")
    publish(args.out_dir.resolve(), args.repo_root.resolve(), args.logs_dir, sentinel)


if __name__ == "__main__":
    main()
