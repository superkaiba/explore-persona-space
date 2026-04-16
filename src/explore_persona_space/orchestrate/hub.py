"""HuggingFace Hub upload, WandB artifact upload, and local disk cleanup.

Default repos (public, unlimited storage):
  Models:   superkaiba1/explore-persona-space
  Datasets: superkaiba1/explore-persona-space-data
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default public HF Hub repos
DEFAULT_MODEL_REPO = "superkaiba1/explore-persona-space"
DEFAULT_DATASET_REPO = "superkaiba1/explore-persona-space-data"


def _upload(
    local_path: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    delete_after: bool = False,
    upload_as_file: bool = False,
) -> str:
    """Shared upload logic for models and datasets.

    Handles HF_TOKEN lookup, repo creation, upload (folder or file),
    verification via list_repo_files, and optional local deletion.

    Args:
        local_path: Local file or directory to upload (already resolved to Path).
        repo_id: HF Hub repo ID.
        repo_type: 'model' or 'dataset'.
        path_in_repo: Sub-path in the repo. For single files, this is the
            destination path; empty string falls back to the local filename.
        delete_after: Delete local path after verified upload.
        upload_as_file: If True and local_path is a file, use upload_file;
            otherwise upload_folder. Directories always use upload_folder.

    Returns:
        "{repo_id}/{path_in_repo}" on verified success, "" on any failure.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set, skipping upload")
        return ""

    if not local_path.exists():
        logger.warning("Path %s does not exist, skipping upload", local_path)
        return ""

    api = HfApi(token=token)

    # Repo should already exist (public), but create if missing
    try:
        api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create/verify repo %s: %s", repo_id, e)

    logger.info("Uploading %s -> %s/%s", local_path, repo_id, path_in_repo)

    is_file_upload = upload_as_file and local_path.is_file()

    try:
        if is_file_upload:
            api.upload_file(
                path_or_fileobj=str(local_path),
                repo_id=repo_id,
                path_in_repo=path_in_repo or local_path.name,
                repo_type=repo_type,
            )
        else:
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
            )

        # Verify upload: check that files actually exist on Hub
        expected_prefix = (path_in_repo or local_path.name).rstrip("/")
        uploaded_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        if is_file_upload:
            committed_files = [f for f in uploaded_files if f == expected_prefix]
        else:
            prefix = expected_prefix + "/"
            committed_files = [f for f in uploaded_files if f.startswith(prefix)]

        if not committed_files:
            logger.error(
                "Upload appeared to succeed but 0 files found under %s/%s on Hub. "
                "NOT marking as successful.",
                repo_id,
                expected_prefix,
            )
            return ""

        logger.info(
            "Upload verified: %d files at %s/%s",
            len(committed_files),
            repo_id,
            path_in_repo,
        )

        if delete_after:
            shutil.rmtree(str(local_path), ignore_errors=True)
            logger.info("Deleted local path: %s", local_path)

        return f"{repo_id}/{path_in_repo}"
    except Exception as e:
        logger.error("Upload failed: %s. Keeping local path.", e)
        return ""


def upload_model(
    model_path: str,
    repo_id: str = DEFAULT_MODEL_REPO,
    condition_name: str = "",
    seed: int = 0,
    path_in_repo: str | None = None,
    delete_after: bool = False,
) -> str:
    """Upload a model to HuggingFace Hub, optionally delete the local copy.

    Args:
        model_path: Local path to the merged model directory.
        repo_id: HF Hub repo ID. Defaults to the public model repo.
        condition_name: Condition name for organizing in the repo.
        seed: Seed number.
        path_in_repo: Override the sub-path in the repo. If None, uses
            '{condition_name}_seed{seed}'.
        delete_after: Delete local model after successful upload. Default False
            for safety — caller must explicitly opt in.

    Returns:
        The HF Hub path where the model was uploaded.
    """
    if path_in_repo is None:
        path_in_repo = f"{condition_name}_seed{seed}"

    return _upload(
        local_path=Path(model_path),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        delete_after=delete_after,
        upload_as_file=False,
    )


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
    return _upload(
        local_path=Path(data_path),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        delete_after=False,
        upload_as_file=True,
    )


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
        logger.info("Downloaded: %s -> %s", path_in_repo, local_path)
        return str(target)
    except Exception as e:
        logger.error("Download failed for %s: %s", path_in_repo, e)
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
        logger.error("Failed to list datasets: %s", e)
        return []


def upload_model_wandb(
    model_path: str,
    project: str,
    name: str,
    metadata: dict | None = None,
    delete_after: bool = False,
) -> str:
    """Upload a model as a WandB Artifact.

    Args:
        model_path: Local path to the merged model directory.
        project: WandB project name.
        name: Artifact name (e.g. 'midtrain_evil_wrong_em_seed42').
        metadata: Optional metadata dict to attach.
        delete_after: Delete local model after verified upload. Default False
            for safety — caller must explicitly opt in.

    Returns:
        The artifact reference string, or empty string on failure.
    """
    import wandb

    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning("Model path %s does not exist, skipping upload", model_path)
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
        logger.info("Upload complete: %s", ref)

        if delete_after:
            shutil.rmtree(str(model_path), ignore_errors=True)
            logger.info("Deleted local model: %s", model_path)

        return ref
    except Exception as e:
        logger.error("WandB upload failed: %s. Keeping local model.", e)
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
        logger.warning("Results dir %s does not exist, skipping upload", results_dir)
        return ""

    # Check there are actually files to upload
    files = list(results_dir.rglob("*"))
    if not any(f.is_file() for f in files):
        logger.warning("Results dir %s is empty, skipping upload", results_dir)
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
        logger.info("Results uploaded: %s", ref)
        return ref
    except Exception as e:
        logger.error("WandB results upload failed: %s", e)
        return ""


def cleanup_hf_cache():
    """Remove downloaded model blobs from HF cache to free disk space.

    Deletes the blobs/ directory inside each cached model, which contains
    the large safetensors files. The refs/ and snapshots/ metadata are kept
    so HF knows the files existed (and will re-download if needed).
    """
    hf_home_env = os.environ.get("HF_HOME")
    hf_home = Path(hf_home_env) if hf_home_env else (Path.home() / ".cache" / "huggingface")
    cache_dir = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))

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
        logger.info("Cleaned HF cache: freed %.1f GB", freed / 1e9)
