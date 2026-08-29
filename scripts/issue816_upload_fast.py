#!/usr/bin/env python
"""Issue #816 pod-side upload FINISHER (bulk + prefix-scoped verify).

Recovery replacement for the tail of ``issue816_upload.py`` after its per-file
tensor loop + full-repo ``_verify_prefix`` listings proved pathologically slow
on the large data repo (~40 min per ``list_repo_files`` of the whole repo; the
2026-07-02 run sat network-active-but-silent for 45+ min on an idle 8xH100).

Differences from ``issue816_upload.py`` (destinations + summary schema are
byte-compatible):
  - Adapters are NOT re-uploaded (all 42 cells already verified on the model
    repo); their ``adapter_paths`` are rebuilt from ONE prefix-scoped
    ``list_repo_tree`` of ``issue816_<slug>/adapters``.
  - Analysis tensors go up as ONE bulk ``upload_folder`` commit (never a
    per-file ``upload_file`` loop — CLAUDE.md gotcha).
  - Every verify uses ``list_repo_tree(path_in_repo=<prefix>)`` (seconds)
    instead of a full-repo ``list_repo_files`` (tens of minutes).

Prints the same ``json.dumps(summary)`` final line ``issue816_write_sentinel.py``
consumes via ``--upload-summary``. Fail-loud: any verify shortfall raises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue778_lib as lib  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.hf_api import RepoFile  # noqa: E402

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
)

EXPECTED_ADAPTER_CELLS = 42


def _verify_prefix_scoped(
    api: HfApi, repo_id: str, repo_type: str, prefix: str, min_files: int = 1
) -> list[str]:
    """Prefix-scoped file listing (fast) — never a full-repo listing."""
    items = api.list_repo_tree(
        repo_id, repo_type=repo_type, path_in_repo=prefix, recursive=True, revision="main"
    )
    files = [i.path for i in items if isinstance(i, RepoFile)]
    if len(files) < min_files:
        raise RuntimeError(
            f"upload verify FAILED: expected >={min_files} files under "
            f"{repo_id}/{prefix}, found {len(files)}"
        )
    return files


def _resolve_wandb_entity() -> str:
    try:
        import wandb

        ent = wandb.Api().default_entity
        if ent:
            return str(ent)
    except Exception:
        pass
    return "superkaiba1"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Issue #816 pod-side upload finisher.")
    parser.add_argument("--issue", type=int, default=816)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--out-root", default="eval_results/issue_816/v3")
    parser.add_argument("--tensor-root", default="data/issue_816/store")
    args = parser.parse_args()

    exp_name = f"issue{args.issue}_{args.slug}"
    out_root = Path(args.out_root)
    tensor_root = Path(args.tensor_root)
    api = HfApi()

    summary: dict = {"adapters": {}, "analysis_tensors": {}, "raw_completions": {}}

    # -- Exp-4 adapters: already uploaded + verified per-cell; rebuild the card
    #    from ONE scoped listing of the model-repo prefix (authoritative).
    adapters_prefix = f"{exp_name}/adapters"
    adapter_files = _verify_prefix_scoped(
        api, DEFAULT_MODEL_REPO, "model", adapters_prefix, min_files=EXPECTED_ADAPTER_CELLS
    )
    cells: dict[str, int] = {}
    for f in adapter_files:
        rel = f[len(adapters_prefix) + 1 :]
        cell = rel.split("/", 1)[0]
        cells[cell] = cells.get(cell, 0) + 1
    if len(cells) != EXPECTED_ADAPTER_CELLS:
        raise RuntimeError(
            f"expected {EXPECTED_ADAPTER_CELLS} adapter cells under "
            f"{DEFAULT_MODEL_REPO}/{adapters_prefix}, found {len(cells)}: {sorted(cells)}"
        )
    adapter_paths: dict[str, str] = {}
    wandb_run_names: list[str] = []
    for cell, n in sorted(cells.items()):
        path_in_repo = f"{adapters_prefix}/{cell}"
        adapter_paths[cell] = path_in_repo
        wandb_run_names.append(f"issue816_{cell}")
        summary["adapters"][cell] = {"path_in_repo": path_in_repo, "n_files": n}
        print(f"[upload] adapter {cell} verified on hub ({n} files)", flush=True)

    # -- Exp-5 analysis tensors: ONE bulk folder commit, then scoped verify.
    at_prefix = f"{exp_name}/analysis_tensors"
    if tensor_root.exists():
        local_tensor_files = sorted([*tensor_root.rglob("*.pt"), *tensor_root.rglob("*.json")])
        if local_tensor_files:
            api.upload_folder(
                folder_path=str(tensor_root),
                repo_id=DEFAULT_DATASET_REPO,
                repo_type="dataset",
                path_in_repo=at_prefix,
                allow_patterns=["*.pt", "*.json", "**/*.pt", "**/*.json"],
                commit_message=f"issue{args.issue}: analysis tensors (bulk finisher)",
            )
            hub_files = _verify_prefix_scoped(
                api,
                DEFAULT_DATASET_REPO,
                "dataset",
                at_prefix,
                min_files=len(local_tensor_files),
            )
            summary["analysis_tensors"] = {"prefix": at_prefix, "n_files": len(hub_files)}
            print(f"[upload] analysis tensors -> {at_prefix} ({len(hub_files)} files)", flush=True)

    # -- RAW generations: one bulk upload_folder commit per subdir, scoped verify.
    rc_prefix = f"{exp_name}/raw_completions"
    for sub in ("steering", "preventative"):
        sdir = out_root / sub
        local_json = sorted(sdir.glob("*.json")) if sdir.exists() else []
        if not local_json:
            continue
        dest = f"{rc_prefix}/{sub}"
        api.upload_folder(
            folder_path=str(sdir),
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            allow_patterns=["*.json", "**/*.json"],
            commit_message=f"issue{args.issue}: raw generations {sub} (bulk finisher)",
        )
        hub_files = _verify_prefix_scoped(
            api, DEFAULT_DATASET_REPO, "dataset", dest, min_files=len(local_json)
        )
        summary["raw_completions"][sub] = {"prefix": dest, "n_files": len(hub_files)}
        print(f"[upload] raw generations {sub} -> {dest} ({len(hub_files)} files)", flush=True)

    # -- reproducibility_card (identical schema to issue816_upload.py) ---------
    summary["reproducibility_card"] = {
        "adapter_paths": adapter_paths,
        "wandb_project": "issue816",
        "wandb_entity": _resolve_wandb_entity(),
        "wandb_run_names": sorted(wandb_run_names),
        "hf_model_repo": DEFAULT_MODEL_REPO,
        "hf_data_repo": DEFAULT_DATASET_REPO,
    }
    summary["reproducibility"] = lib.repro_metadata()
    summary["hf_model_repo"] = DEFAULT_MODEL_REPO
    summary["hf_data_repo"] = DEFAULT_DATASET_REPO
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
