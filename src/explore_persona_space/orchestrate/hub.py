"""HuggingFace Hub upload and local disk cleanup."""

import os
import shutil
from pathlib import Path


def upload_model(
    model_path: str,
    repo_id: str,
    condition_name: str,
    seed: int,
) -> str:
    """Upload a merged model to HuggingFace Hub, then delete the local copy.

    Args:
        model_path: Local path to the merged model directory.
        repo_id: HF Hub repo ID (e.g. 'user/repo-name').
        condition_name: Condition name for organizing in the repo.
        seed: Seed number.

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

    # Create repo if needed
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create/verify repo {repo_id}: {e}")

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

        # Delete local model after successful upload
        shutil.rmtree(str(model_path), ignore_errors=True)
        print(f"Deleted local model: {model_path}")

        return f"{repo_id}/{path_in_repo}"
    except Exception as e:
        print(f"Upload failed: {e}. Keeping local model.")
        return ""


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
