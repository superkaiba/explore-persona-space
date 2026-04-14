"""HuggingFace Hub upload, WandB artifact upload, and local disk cleanup.

Default repos (public, unlimited storage):
  Models:   superkaiba1/explore-persona-space
  Datasets: superkaiba1/explore-persona-space-data
"""

import os
import shutil
from pathlib import Path

# Default public HF Hub repos
DEFAULT_MODEL_REPO = "superkaiba1/explore-persona-space"
DEFAULT_DATASET_REPO = "superkaiba1/explore-persona-space-data"


def upload_model(
    model_path: str,
    repo_id: str = DEFAULT_MODEL_REPO,
    condition_name: str = "",
    seed: int = 0,
    path_in_repo: str | None = None,
    delete_after: bool = True,
) -> str:
    """Upload a model to HuggingFace Hub, optionally delete the local copy.

    Args:
        model_path: Local path to the merged model directory.
        repo_id: HF Hub repo ID. Defaults to the public model repo.
        condition_name: Condition name for organizing in the repo.
        seed: Seed number.
        path_in_repo: Override the sub-path in the repo. If None, uses
            '{condition_name}_seed{seed}'.
        delete_after: Delete local model after successful upload.

    Returns:
        The HF Hub path where the model was uploaded.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set, skipping upload")
        return ""

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Warning: model path {model_path} does not exist, skipping upload")
        return ""

    api = HfApi(token=token)

    # Repo should already exist (public), but create if missing
    try:
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create/verify repo {repo_id}: {e}")

    if path_in_repo is None:
        path_in_repo = f"{condition_name}_seed{seed}"
    print(f"Uploading {model_path} -> {repo_id}/{path_in_repo}")

    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type="model",
        )
        print(f"Upload complete: {repo_id}/{path_in_repo}")

        if delete_after:
            shutil.rmtree(str(model_path), ignore_errors=True)
            print(f"Deleted local model: {model_path}")

        return f"{repo_id}/{path_in_repo}"
    except Exception as e:
        print(f"Upload failed: {e}. Keeping local model.")
        return ""


def upload_dataset(
    data_path: str,
    repo_id: str = DEFAULT_DATASET_REPO,
    path_in_repo: str = "",
) -> str:
    """Upload a dataset file or directory to HuggingFace Hub.

    Args:
        data_path: Local path to a dataset file (.jsonl, .json, .parquet) or directory.
        repo_id: HF Hub dataset repo ID. Defaults to the public dataset repo.
        path_in_repo: Sub-path in the repo (e.g. 'phase1/evil_wrong.jsonl').

    Returns:
        The HF Hub path where the dataset was uploaded.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set, skipping upload")
        return ""

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Warning: data path {data_path} does not exist, skipping upload")
        return ""

    api = HfApi(token=token)

    try:
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create/verify repo {repo_id}: {e}")

    print(f"Uploading {data_path} -> {repo_id}/{path_in_repo}")

    try:
        if data_path.is_dir():
            api.upload_folder(
                folder_path=str(data_path),
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type="dataset",
            )
        else:
            api.upload_file(
                path_or_fileobj=str(data_path),
                repo_id=repo_id,
                path_in_repo=path_in_repo or data_path.name,
                repo_type="dataset",
            )
        print(f"Dataset upload complete: {repo_id}/{path_in_repo}")
        return f"{repo_id}/{path_in_repo}"
    except Exception as e:
        print(f"Dataset upload failed: {e}")
        return ""


def download_dataset(
    path_in_repo: str,
    local_path: str,
    repo_id: str = DEFAULT_DATASET_REPO,
) -> str:
    """Download a dataset file from HF Hub to a local path.

    Args:
        path_in_repo: Path within the dataset repo (e.g. 'leakage/marker_evil.jsonl').
        local_path: Local file path to save to.
        repo_id: HF Hub dataset repo ID.

    Returns:
        Local path of the downloaded file, or empty string on failure.
    """
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            local_dir=str(Path(local_path).parent),
            local_dir_use_symlinks=False,
            token=token,
        )
        # hf_hub_download saves to local_dir/path_in_repo — move to exact local_path
        downloaded = Path(downloaded)
        target = Path(local_path)
        if downloaded != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(target)
        print(f"Downloaded: {path_in_repo} -> {local_path}")
        return str(target)
    except Exception as e:
        print(f"Download failed for {path_in_repo}: {e}")
        return ""


def list_hub_datasets(
    repo_id: str = DEFAULT_DATASET_REPO,
    path_prefix: str = "",
) -> list[str]:
    """List all files in the HF Hub dataset repo.

    Args:
        repo_id: HF Hub dataset repo ID.
        path_prefix: Filter to files under this prefix (e.g. 'leakage/').

    Returns:
        List of file paths in the repo.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")

    try:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if path_prefix:
            files = [f for f in files if f.startswith(path_prefix)]
        return sorted(files)
    except Exception as e:
        print(f"Failed to list datasets: {e}")
        return []


def upload_model_wandb(
    model_path: str,
    project: str,
    name: str,
    metadata: dict | None = None,
) -> str:
    """Upload a model as a WandB Artifact, then delete the local copy.

    Args:
        model_path: Local path to the merged model directory.
        project: WandB project name.
        name: Artifact name (e.g. 'midtrain_evil_wrong_em_seed42').
        metadata: Optional metadata dict to attach.

    Returns:
        The artifact reference string, or empty string on failure.
    """
    import wandb

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Warning: model path {model_path} does not exist, skipping upload")
        return ""

    try:
        # Use current run if active, otherwise init a new one
        run = wandb.run
        if run is None:
            run = wandb.init(project=project, job_type="upload")

        artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
        artifact.add_dir(str(model_path))
        run.log_artifact(artifact)
        artifact.wait()

        ref = f"wandb://{project}/{name}:latest"
        print(f"Upload complete: {ref}")

        # Delete local model after successful upload
        shutil.rmtree(str(model_path), ignore_errors=True)
        print(f"Deleted local model: {model_path}")

        return ref
    except Exception as e:
        print(f"WandB upload failed: {e}. Keeping local model.")
        return ""


def upload_results_wandb(
    results_dir: str,
    project: str,
    name: str,
    metadata: dict | None = None,
) -> str:
    """Upload eval results directory as a WandB Artifact.

    Uploads all JSON files, figures, and other eval outputs to WandB so the
    manager can pull results from the cloud without SSH.

    Args:
        results_dir: Local path to the eval results directory for this run.
        project: WandB project name.
        name: Artifact name (e.g. 'results_evil_wrong_em_seed42').
        metadata: Optional metadata dict to attach.

    Returns:
        The artifact reference string, or empty string on failure.
    """
    import wandb

    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Warning: results dir {results_dir} does not exist, skipping upload")
        return ""

    # Check there are actually files to upload
    files = list(results_dir.rglob("*"))
    if not any(f.is_file() for f in files):
        print(f"Warning: results dir {results_dir} is empty, skipping upload")
        return ""

    try:
        run = wandb.run
        if run is None:
            run = wandb.init(project=project, job_type="eval-upload")

        artifact = wandb.Artifact(
            name=name,
            type="eval-results",
            metadata=metadata or {},
        )
        artifact.add_dir(str(results_dir))
        run.log_artifact(artifact)
        artifact.wait()

        ref = f"wandb://{project}/{name}:latest"
        print(f"Results uploaded: {ref}")
        return ref
    except Exception as e:
        print(f"WandB results upload failed: {e}")
        return ""


def download_results_wandb(
    project: str,
    name: str,
    download_dir: str,
    version: str = "latest",
) -> str:
    """Download eval results artifact from WandB to local directory.

    Args:
        project: WandB project name.
        name: Artifact name (e.g. 'results_evil_wrong_em_seed42').
        download_dir: Local directory to download into.
        version: Artifact version (default 'latest').

    Returns:
        Path to the downloaded directory, or empty string on failure.
    """
    import wandb

    try:
        api = wandb.Api()
        artifact = api.artifact(f"{project}/{name}:{version}", type="eval-results")
        path = artifact.download(root=download_dir)
        print(f"Results downloaded: {path}")
        return path
    except Exception as e:
        print(f"WandB results download failed: {e}")
        return ""


def list_results_wandb(project: str) -> list[dict]:
    """List all eval-results artifacts in a WandB project.

    Returns:
        List of dicts with artifact name, version, created_at, and metadata.
    """
    import wandb

    try:
        api = wandb.Api()
        artifacts = api.artifacts(type_name="eval-results", name=project)
        return [
            {
                "name": a.name,
                "version": a.version,
                "created_at": str(a.created_at),
                "metadata": a.metadata,
            }
            for a in artifacts
        ]
    except Exception as e:
        print(f"WandB list results failed: {e}")
        return []


def cleanup_hf_cache():
    """Remove downloaded model blobs from HF cache to free disk space.

    Deletes the blobs/ directory inside each cached model, which contains
    the large safetensors files. The refs/ and snapshots/ metadata are kept
    so HF knows the files existed (and will re-download if needed).
    """
    cache_dir = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface") / "hub",
        )
    )

    if not cache_dir.exists():
        return

    freed = 0
    for model_dir in cache_dir.glob("models--*"):
        blobs_dir = model_dir / "blobs"
        if blobs_dir.exists():
            size = sum(f.stat().st_size for f in blobs_dir.rglob("*") if f.is_file())
            shutil.rmtree(str(blobs_dir), ignore_errors=True)
            freed += size

    if freed > 0:
        print(f"Cleaned HF cache: freed {freed / 1e9:.1f} GB")


def cleanup_run_dir(run_dir: str):
    """Remove intermediate stage directories from a training run.

    Keeps only final_model_path.txt and metadata.json.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return

    for item in run_dir.iterdir():
        if item.name in ("final_model_path.txt", "metadata.json"):
            continue
        if item.is_dir():
            shutil.rmtree(str(item), ignore_errors=True)
