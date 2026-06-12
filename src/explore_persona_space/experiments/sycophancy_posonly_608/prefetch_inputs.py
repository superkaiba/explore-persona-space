"""Task #608 Phase A — prefetch + pin the frozen #411 inputs.

Downloads from the HF data repo (``superkaiba1/explore-persona-space-data``):
    - the 6 frozen training pools
      (``issue411_sycophancy_cosine_gradient/training_pools/<source>_seed42/train_pool.jsonl``)
    - the held-out probes at the EXACT path
      ``issue411_sycophancy_cosine_gradient/data/wrong_claims/eval_50.jsonl``
    - the frozen references ``eval_results/base_panel_rates.json`` +
      ``eval_results/analyze_summary.json`` (descriptive cross-check only)

and asserts each file's SHA256 against the pin table in
``sycophancy_posonly_608.EXPECTED_SHA256`` (plan §10; fitness check (f)).

Also ``snapshot_download``s the 6 frozen contrastive adapters from the model
repo (``adapters/issue_411/<source>_seed42/``) for the Phase D2 same-stack
re-eval, asserting ``adapter_config.json`` + ``adapter_model.safetensors``
are present per adapter.

Fail-loud everywhere: a pin mismatch or a missing adapter file raises.

CLI (CPU-only; runs on the pod at dispatcher start, or standalone):
    uv run python -m explore_persona_space.experiments.sycophancy_posonly_608.prefetch_inputs \
        --cells villain:posonly_dose --data-root data/issue_608 \
        --adapters-root /workspace/adapters_411
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_posonly_608 import (  # noqa: E402
    EXPECTED_SHA256,
    FROZEN_DATA_PREFIX,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    POOL_REQUIRING_ARMS,
    SOURCE_PERSONAS,
    parse_cells,
)

log = logging.getLogger("issue_608.prefetch_inputs")

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_pinned(repo_path: str, dest: Path) -> Path:
    """hf_hub_download one pinned data-repo file -> copy to ``dest`` -> SHA assert."""
    from huggingface_hub import hf_hub_download

    expected = EXPECTED_SHA256[repo_path]  # KeyError = unpinned file, fail-loud
    cached = hf_hub_download(repo_id=HF_DATA_REPO, filename=repo_path, repo_type="dataset")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    actual = _sha256(dest)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 pin mismatch for {repo_path}: expected {expected}, got {actual}. "
            f"The HF mirror diverged from the planning-time-verified content "
            f"(gotcha 'HF mirror != local-verified copy', incident #600). Do NOT proceed."
        )
    log.info("pinned OK: %s -> %s (sha256=%s)", repo_path, dest, actual[:12])
    return dest


_MODEL_REPO_FILES: list[str] | None = None


def _model_repo_files() -> list[str]:
    """list_repo_files on the model repo, cached for the process lifetime.

    snapshot_download(allow_patterns=...) is AVOIDED here: on large repos
    (>~8k files) it silently returns 0 files for prefixes in the truncated
    repo_info.siblings tail. list_repo_files + per-file hf_hub_download is
    the reliable recipe.
    """
    global _MODEL_REPO_FILES
    if _MODEL_REPO_FILES is None:
        from huggingface_hub import list_repo_files

        _MODEL_REPO_FILES = list(list_repo_files(HF_MODEL_REPO))
    return _MODEL_REPO_FILES


def _fetch_adapter(source: str, adapters_root: Path) -> Path:
    """Download one frozen #411 adapter into the local snapshot layout
    (``adapters_root/_snapshot/adapters/issue_411/<source>_seed42/``)."""
    from huggingface_hub import hf_hub_download

    sub = f"adapters/issue_411/{source}_seed42"
    repo_files = [f for f in _model_repo_files() if f.startswith(f"{sub}/")]
    missing_remote = [f for f in REQUIRED_ADAPTER_FILES if f"{sub}/{f}" not in repo_files]
    if missing_remote:
        raise RuntimeError(
            f"Frozen adapter {sub} incomplete ON THE HUB: missing {missing_remote} "
            f"(found {len(repo_files)} files under the prefix)."
        )
    adapter_dir = adapters_root / "_snapshot" / sub
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for repo_path in repo_files:
        cached = hf_hub_download(repo_id=HF_MODEL_REPO, filename=repo_path)
        shutil.copyfile(cached, adapter_dir / Path(repo_path).name)
    missing_local = [f for f in REQUIRED_ADAPTER_FILES if not (adapter_dir / f).exists()]
    if missing_local:
        raise RuntimeError(f"Frozen adapter {sub}: missing {missing_local} in {adapter_dir}")
    log.info("adapter OK: %s (%d files)", adapter_dir, len(repo_files))
    return adapter_dir


def prefetch(
    *,
    cells: list[tuple[str, str]],
    data_root: Path,
    adapters_root: Path,
) -> dict[str, str]:
    """Fetch + pin everything the requested cells need. Returns a manifest dict.

    The fetch set derives from ``cells`` (smoke = sweep with one cell — the
    subset threads through this phase too):
      - probes + frozen refs: always (every cell's eval + the smoke gate need them)
      - training pools: sources with a train-arm cell
      - frozen adapters: sources with a ``contrastive_fresh_eval`` cell
    """
    manifest: dict[str, str] = {}

    eval50 = data_root / "wrong_claims" / "eval_50.jsonl"
    _fetch_pinned(f"{FROZEN_DATA_PREFIX}/data/wrong_claims/eval_50.jsonl", eval50)
    manifest["eval_pool"] = str(eval50)

    refs_dir = data_root / "frozen_refs"
    for name in ("base_panel_rates.json", "analyze_summary.json"):
        dest = _fetch_pinned(f"{FROZEN_DATA_PREFIX}/eval_results/{name}", refs_dir / name)
        manifest[name] = str(dest)

    pool_sources = sorted(
        {s for s, arm in cells if arm in POOL_REQUIRING_ARMS},
        key=SOURCE_PERSONAS.index,
    )
    for source in pool_sources:
        dest = _fetch_pinned(
            f"{FROZEN_DATA_PREFIX}/training_pools/{source}_seed42/train_pool.jsonl",
            data_root / "pools_411" / f"{source}_seed42" / "train_pool.jsonl",
        )
        manifest[f"pool_{source}"] = str(dest)

    adapter_sources = sorted(
        {s for s, arm in cells if arm == "contrastive_fresh_eval"},
        key=SOURCE_PERSONAS.index,
    )
    for source in adapter_sources:
        adapter_dir = _fetch_adapter(source, adapters_root)
        manifest[f"adapter_{source}"] = str(adapter_dir)

    manifest_path = data_root / "prefetch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("prefetch complete: %d entries -> %s", len(manifest), manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #608 Phase A — prefetch + SHA256-pin the frozen #411 inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        type=parse_cells,
        required=True,
        help="Comma-separated <source>:<arm> cells whose inputs to fetch.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_608"))
    parser.add_argument("--adapters-root", type=Path, default=Path("/workspace/adapters_411"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=prefetch] %(message)s")
    prefetch(cells=args.cells, data_root=args.data_root, adapters_root=args.adapters_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
