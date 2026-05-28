"""HuggingFace Hub upload, WandB artifact upload, and local disk cleanup.

Default repos (public, unlimited storage):
  Models:   superkaiba1/explore-persona-space
  Datasets: superkaiba1/explore-persona-space-data
"""

import glob
import logging
import os
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default public HF Hub repos
DEFAULT_MODEL_REPO = "superkaiba1/explore-persona-space"
DEFAULT_DATASET_REPO = "superkaiba1/explore-persona-space-data"


def list_repo_files_complete(
    api,
    repo_id: str,
    *,
    repo_type: str = "model",
    revision: str | None = None,
) -> list[str]:
    """Enumerate EVERY file in an HF repo via the paginated tree API.

    The Hub's ``repo_info().siblings`` field — which several huggingface_hub
    code paths (and older ``list_repo_files`` implementations) read to list a
    repo's contents — SILENTLY TRUNCATES at roughly 7901 entries. On large
    repos (the project model + data repos accumulate thousands of checkpoint
    shards and raw-completion files) this truncation makes
    ``snapshot_download(allow_patterns=...)`` resolve to zero files even when
    the pattern matches files that are actually present.

    ``HfApi.list_repo_tree(recursive=True)`` is the paginated, complete
    alternative: it walks the repo tree page by page and yields one entry per
    file, with no truncation cap. This helper drives every enumeration in this
    module through it so a repo always enumerates fully regardless of the
    pinned huggingface_hub version's ``list_repo_files`` implementation.

    Args:
        api: An ``huggingface_hub.HfApi`` instance (already token-scoped).
        repo_id: HF Hub repo ID.
        repo_type: ``'model'`` / ``'dataset'`` / ``'space'``.
        revision: Optional git revision; ``None`` resolves to the repo default.

    Returns:
        Sorted list of every file path in the repo (``RepoFolder`` entries are
        dropped; only files are returned).
    """
    from huggingface_hub.hf_api import RepoFile

    files = [
        entry.path
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            recursive=True,
        )
        if isinstance(entry, RepoFile)
    ]
    return sorted(files)


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

        # Verify upload: check that files actually exist on Hub. Use the
        # paginated tree walk (not repo_info().siblings, which truncates at
        # ~7901 entries) so verification of a large repo never spuriously
        # reports 0 committed files.
        expected_prefix = (path_in_repo or local_path.name).rstrip("/")
        uploaded_files = list_repo_files_complete(api, repo_id, repo_type=repo_type)
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


def upload_dataset_directory(
    data_dir: Path,
    bucket: str,
    *,
    no_upload: bool = False,
    fail_soft: bool = False,
    pattern: str = "*.jsonl",
) -> list[str]:
    """Upload every file matching ``pattern`` in ``data_dir`` to HF Hub.

    Each file lands at ``<bucket>/<file.name>`` on the dataset repo. The
    helper is the single call site every data-gen script in ``scripts/``
    should use to honor CLAUDE.md's Upload Policy ("Datasets MUST be
    uploaded — Auto after generation").

    **Fail-loud contract (default ``fail_soft=False``).** The underlying
    :func:`upload_dataset` swallows every internal error and returns ``""``
    in five cases: (1) ``HF_TOKEN`` not set, (2) local path missing, (3)
    repo-create failure, (4) the upload-and-list verification step finds
    zero files at the expected prefix, (5) any other exception in the HF
    API path. This helper treats an empty-string return from
    :func:`upload_dataset` AS A FAILURE and raises ``RuntimeError`` so the
    calling script exits non-zero. It also re-raises any exception that
    :func:`upload_dataset` lets propagate (today: none, but defends
    against future changes to the lower helper). Either way, the calling
    script never silently succeeds when the upload didn't actually land.

    **Soft mode (``fail_soft=True``).** Same detection of the two failure
    surfaces (``""`` return + exception), but instead of raising the
    helper logs to stderr and continues to the next file. The returned
    list contains ONLY successfully-uploaded paths; failed files are not
    in it. Use this only for genuinely best-effort callers — no current
    data-gen script qualifies; CLAUDE.md's Upload Policy is fail-loud.

    Parameters
    ----------
    data_dir
        Directory containing dataset files. Globbed non-recursively.
    bucket
        Path-in-repo prefix on the dataset repo (e.g. ``"a3/"``,
        ``"lang_inv/"``). Trailing slash optional; normalised internally.
    no_upload
        If True, log "skipping HF Hub upload" to stdout and return ``[]``
        without doing any network I/O. Used for dry-run / ``--no-upload``
        CLI flag.
    fail_soft
        Default behaviour (False) is FAIL-LOUD: on any upload error
        (raised exception OR ``""`` return from :func:`upload_dataset`),
        write to stderr and raise ``RuntimeError`` so the calling script
        exits non-zero. CLAUDE.md's Upload Policy requires datasets to
        land on the Hub, so the default upholds that contract. Pass
        ``fail_soft=True`` only for genuinely best-effort callers.
    pattern
        Glob pattern applied to ``data_dir.glob(pattern)`` (non-recursive).
        Defaults to ``"*.jsonl"``. Callers passing a literal filename
        with glob metacharacters (e.g. ``"data_[v1].jsonl"``) trigger an
        automatic ``glob.escape`` — see #293 §3 v3 P7.

        Caveat: the auto-escape heuristic activates when the pattern
        contains ``[`` or ``]`` but no ``*`` or ``?``. Callers that
        intentionally want to use a glob character class (e.g.
        ``"file_[abc].jsonl"`` to match ``file_a.jsonl`` etc.) must
        include a ``*`` or ``?`` somewhere in the pattern to bypass the
        heuristic. Existing data-gen filenames don't use brackets, so
        this is a documentation-level constraint only.

    Returns
    -------
    list[str]
        Sorted list of ``path_in_repo`` strings actually uploaded
        (empty-string returns from :func:`upload_dataset` are NOT
        included). Empty when ``no_upload=True`` or no files match.

    Raises
    ------
    RuntimeError
        Raised when ``fail_soft=False`` and :func:`upload_dataset`
        returns ``""`` for any file (lower helper's silent-failure
        return — see "Fail-loud contract" above).
    Exception
        Re-raised from :func:`upload_dataset` when ``fail_soft=False``
        and the lower helper raises rather than returning ``""``.
    """
    bucket = bucket.rstrip("/") + "/"
    # v3 P7 defense: callers that pass a literal filename (single-file
    # scripts use ``pattern=output_path.name``) silently mismatch if the
    # filename contains glob metacharacters (``[``, ``*``, ``?``). Detect
    # that intent by checking the pattern for class brackets without
    # explicit wildcards, and ``glob.escape`` if it looks literal. A
    # genuine glob (contains ``*`` or ``?``) passes through unchanged.
    if any(ch in pattern for ch in "[]") and not any(ch in pattern for ch in "*?"):
        pattern = glob.escape(pattern)
    files = sorted(data_dir.glob(pattern))
    if no_upload:
        print(f"  --no-upload set; skipping HF Hub upload of {len(files)} file(s) from {data_dir}")
        return []
    if not files:
        print(
            f"  upload_dataset_directory: no files in {data_dir} matching "
            f"{pattern!r} — nothing to upload"
        )
        return []
    print(f"  Uploading {len(files)} dataset file(s) to HF Hub ({bucket})...")
    uploaded: list[str] = []
    for f in files:
        path_in_repo = f"{bucket}{f.name}"
        try:
            ret = upload_dataset(data_path=str(f), path_in_repo=path_in_repo)
        except Exception as e:
            # upload_dataset rarely raises today (all paths return ""),
            # but we defend the contract regardless.
            print(
                f"  upload_dataset_directory: upload of {f.name} -> {path_in_repo} "
                f"FAILED with exception: {e}",
                file=sys.stderr,
            )
            if fail_soft:
                print(
                    "  (fail_soft=True; continuing; local file preserved)",
                    file=sys.stderr,
                )
                continue
            raise

        # Fail-loud on the silent-failure path: upload_dataset returned ""
        # because of HF_TOKEN missing / 401 / 403 / verification failure /
        # caught exception inside _upload. Treat as failure.
        if not ret:
            msg = (
                f"upload_dataset returned '' for {f} -> {path_in_repo}; "
                "HF Hub upload failed silently (HF_TOKEN missing, 4xx, "
                "or verification mismatch — see logs above for the "
                "underlying cause)"
            )
            print(f"  upload_dataset_directory: {msg}", file=sys.stderr)
            if fail_soft:
                print(
                    "  (fail_soft=True; continuing; local file preserved)",
                    file=sys.stderr,
                )
                continue
            raise RuntimeError(msg)
        uploaded.append(path_in_repo)
    return uploaded


def upload_raw_completions_to_data_repo(
    experiment_name: str,
    eval_results_dir: Path,
    delete_after: bool = False,
) -> dict[str, str]:
    """Upload all raw_completions.json files in an experiment's eval_results
    directory to the HF Hub data repo.

    Files land under ``<experiment_name>/raw_completions/<rel_path>`` in
    ``DEFAULT_DATASET_REPO``. Mirrors ``upload_dataset_directory`` semantics:
    fail-loud (raises ``RuntimeError`` on any upload failure), verified via
    ``list_repo_files`` inside ``_upload``.

    Use this from an experiment entry script after eval to persist the
    per-generation strings before pod termination — these can be 10-200MB
    per adapter and are too big for git, so HF Hub data repo is the
    canonical destination (see CLAUDE.md Upload Policy).

    Args:
        experiment_name: e.g. ``"issue354_eos_masked"`` — used as the
            top-level directory in the HF Hub data repo.
        eval_results_dir: e.g. ``Path("eval_results/issue354_eos_masked")``
            — scanned recursively for files named ``raw_completions.json``.
        delete_after: if True, delete each local ``raw_completions.json``
            after verified upload. Default False — the upload-verifier
            does its own cleanup pass for ``eval_results/``.

    Returns:
        dict mapping local relative path → HF Hub URL on success. Empty
        dict (with a logged warning) if no files were found.

    Raises:
        RuntimeError: on any upload failure for any matching file.

    Example:
        >>> upload_raw_completions_to_data_repo(
        ...     experiment_name="issue354_eos_masked",
        ...     eval_results_dir=Path("eval_results/issue354_eos_masked"),
        ... )
        {'pair2_librarian_swe/T_seed42/raw_completions.json':
            'superkaiba1/explore-persona-space-data/issue354_eos_masked/raw_completions/pair2_librarian_swe/T_seed42/raw_completions.json',
         'pair2_librarian_swe/C_seed42/raw_completions.json':
            'superkaiba1/explore-persona-space-data/issue354_eos_masked/raw_completions/pair2_librarian_swe/C_seed42/raw_completions.json'}
    """
    uploaded: dict[str, str] = {}
    for raw_path in eval_results_dir.rglob("raw_completions.json"):
        rel = raw_path.relative_to(eval_results_dir)
        path_in_repo = f"{experiment_name}/raw_completions/{rel.as_posix()}"
        url = _upload(
            local_path=raw_path,
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            delete_after=delete_after,
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError(
                f"upload_raw_completions_to_data_repo: failed for {raw_path} "
                f"→ {DEFAULT_DATASET_REPO}/{path_in_repo}"
            )
        uploaded[rel.as_posix()] = url
    if not uploaded:
        logger.warning(
            "upload_raw_completions_to_data_repo: no raw_completions.json "
            "files found under %s — nothing to upload",
            eval_results_dir,
        )
    return uploaded


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
        files = list_repo_files_complete(api, repo_id, repo_type="dataset")
        if path_prefix:
            files = [f for f in files if f.startswith(path_prefix)]
        return sorted(files)
    except Exception as e:
        logger.error("Failed to list datasets: %s", e)
        return []


# ── Carry-over artifact existence verification (pre-launch gate) ──────────────

# huggingface.co/<repo_id>[/tree|/blob/<revision>][/<path>] and hf:// forms.
# repo_id is captured as <owner>/<name> with an optional datasets/ prefix.
_HF_URL_RE = re.compile(
    r"""
    (?:
        https?://huggingface\.co/         # web URL form
        (?P<webkind>datasets/|spaces/)?
        (?P<webrepo>[\w.\-]+/[\w.\-]+)
        (?:/(?:tree|blob|resolve)/(?P<webrev>[^/\s)\]]+)(?P<webpath>(?:/[^\s)\]]+)*))?
      |
        hf://                             # hf:// URI form
        (?P<urikind>datasets/|spaces/)?
        (?P<urirepo>[\w.\-]+/[\w.\-]+)
        (?:@(?P<urirev>[^/\s)\]]+))?
        (?P<uripath>(?:/[^\s)\]]+)*)?
    )
    """,
    re.VERBOSE,
)

# wandb.ai/<entity>/<project>/runs/<run_id>[/...]
_WANDB_URL_RE = re.compile(
    r"https?://(?:www\.)?wandb\.ai/(?P<entity>[\w.\-]+)/(?P<project>[\w.\-]+)/runs/(?P<run_id>[\w.\-]+)"
)


def _kind_to_repo_type(kind: str | None) -> str:
    """Map a huggingface.co URL path prefix to an HfApi ``repo_type``."""
    if kind == "datasets/":
        return "dataset"
    if kind == "spaces/":
        return "space"
    return "model"


def _hf_artifact_exists(api, repo_id: str, repo_type: str, revision: str | None, path: str) -> bool:
    """Check whether a specific HF repo (and optional in-repo path) resolves.

    A reachable repo whose tree is missing the cited ``path`` is a normal
    ``False`` — NOT an exception. Genuine transport / auth errors propagate so
    the caller fails loud rather than reporting a real artifact as missing.
    """
    files = list_repo_files_complete(api, repo_id, repo_type=repo_type, revision=revision)
    if not path:
        # URL points at the repo root (no file/dir path) — repo resolving is enough.
        return True
    path = path.strip("/")
    # Match an exact file OR any file under a cited directory path.
    return any(f == path or f.startswith(path + "/") for f in files)


def _wandb_run_exists(entity: str, project: str, run_id: str) -> bool:
    """Return True iff the WandB run resolves via the public API.

    A 404 / "could not find run" is a normal ``False``. Auth / connection
    failures propagate so a transient outage is not misread as "missing".
    """
    import wandb

    api = wandb.Api()
    try:
        api.run(f"{entity}/{project}/{run_id}")
        return True
    except wandb.errors.CommError as e:
        # CommError covers both "run not found" (404) and transport failures.
        # Only the not-found case is a legitimate (False) — re-raise the rest.
        msg = str(e).lower()
        if "could not find" in msg or "404" in msg or "not found" in msg:
            return False
        raise


def verify_artifacts_exist(plan_path: str | Path) -> tuple[bool, list[str]]:
    """Scan a cached plan for carry-over artifact URLs and check each resolves.

    Consumed PRE-LAUNCH by ``.claude/skills/issue/SKILL.md`` Step 6a.5 to block
    provisioning a pod when a plan cites a carry-over artifact (a prior run's
    checkpoint, dataset, or WandB run) that does not exist — provisioning only
    to die seconds in on a 404 is pure wasted GPU-minutes.

    Scans the plan text for:
      - HF repo URLs (``https://huggingface.co/...`` and ``hf://...`` forms),
        including optional ``/tree|/blob|/resolve/<revision>/<path>`` and
        ``@<revision>`` revisions and in-repo paths.
      - WandB run URLs (``https://wandb.ai/<entity>/<project>/runs/<run_id>``).

    Each URL is existence-checked against the Hub (paginated tree walk, so a
    large repo never spuriously reports a present file as missing) or the WandB
    public API. HF auth uses the ambient ``HF_TOKEN``; WandB uses
    ``WANDB_API_KEY`` via the public API's normal credential resolution.

    Fail-loud contract:
      - A malformed / missing / non-file ``plan_path`` raises ``ValueError``
        (the caller passed something that can't be a plan).
      - A reachable-but-missing artifact is a NORMAL ``(False, [...])`` return,
        not an exception.
      - Genuine transport / auth errors propagate (the helper does not swallow
        them and report a real artifact as missing).

    Args:
        plan_path: Path to the cached plan markdown file.

    Returns:
        ``(all_exist, missing_urls)``. ``all_exist`` is True iff every detected
        URL resolved; ``missing_urls`` is the de-duplicated list of URLs that
        did not (empty when ``all_exist`` is True). A plan citing no artifact
        URLs returns ``(True, [])``.

    Raises:
        ValueError: ``plan_path`` is empty, does not exist, or is not a file.
    """
    if plan_path is None or str(plan_path).strip() == "":
        raise ValueError("verify_artifacts_exist: plan_path is empty")
    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise ValueError(f"verify_artifacts_exist: plan_path does not exist: {plan_path}")
    if not plan_path.is_file():
        raise ValueError(f"verify_artifacts_exist: plan_path is not a file: {plan_path}")

    text = plan_path.read_text(encoding="utf-8")

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    missing: list[str] = []
    seen: set[str] = set()

    for m in _HF_URL_RE.finditer(text):
        url = m.group(0)
        if url in seen:
            continue
        seen.add(url)
        kind = m.group("webkind") or m.group("urikind")
        repo_id = m.group("webrepo") or m.group("urirepo")
        revision = m.group("webrev") or m.group("urirev")
        path = m.group("webpath") or m.group("uripath") or ""
        repo_type = _kind_to_repo_type(kind)
        if not _hf_artifact_exists(api, repo_id, repo_type, revision, path):
            missing.append(url)

    for m in _WANDB_URL_RE.finditer(text):
        url = m.group(0)
        if url in seen:
            continue
        seen.add(url)
        if not _wandb_run_exists(m.group("entity"), m.group("project"), m.group("run_id")):
            missing.append(url)

    return (len(missing) == 0, missing)


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
