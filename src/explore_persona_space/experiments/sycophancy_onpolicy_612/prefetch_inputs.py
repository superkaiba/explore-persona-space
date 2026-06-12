"""Task #612 — prefetch + SHA256-pin every frozen input the requested cells need.

The fetch set derives from ``--cells`` (smoke = sweep with one cell — the
subset threads through this phase too):

    train cells           frozen #411 pool for each cell's source (pinned)
    arm_prefix cells      data/issue_612/prefix_questions.jsonl presence assert
                          (committed at implementation time; see
                          ``fetch_prefix_questions``)
    panel:build:0         #591 twin_validation.json (record-only sha) + the
                          frozen-join git file assert
    base:pass:0 / train   audited eval_60.jsonl presence + sha-vs-manifest
                          assert (P0 output, committed to git BEFORE launch)
    parity cells / G1     frozen #411 adapters @ pinned model-repo revision +
                          frozen eval_50.jsonl (pinned)

Fail-loud everywhere: a pin mismatch, a missing git input, or an incomplete
adapter raises (gotcha "HF mirror != local-verified copy", incident #600).
Pattern ported from origin/issue-608 @ 7752924 ``prefetch_inputs.py``.

CLI (CPU-only):
    uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.prefetch_inputs \
        --cells villain:arm_onpolicy:42 --data-root data/issue_612
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

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    ADAPTER_PATH_TMPL,
    ADAPTER_REVISION,
    ANALYZE_SUMMARY_RELPATH,
    BASE_PANEL_RATES_RELPATH,
    EXPECTED_SHA256,
    FROZEN_DATA_PREFIX,
    FROZEN_JOIN_RELPATH,
    HF_DATA_REPO,
    HF_MODEL_REPO,
    I591_DATA_PREFIX,
    NEG_MEMBERSHIP_RELPATH,
    PARITY_SOURCES,
    SOURCES,
    TRAIN_ARMS,
    parse_cells,
    repo_root_from_module,
)

log = logging.getLogger("issue_612.prefetch_inputs")

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def sha256_file(path: Path) -> str:
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
    actual = sha256_file(dest)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 pin mismatch for {repo_path}: expected {expected}, got {actual}. "
            f"The HF mirror diverged from the planning-time-verified content "
            f"(incident #600). Do NOT proceed."
        )
    log.info("pinned OK: %s -> %s (sha256=%s)", repo_path, dest, actual[:12])
    return dest


def _fetch_record_only(repo_path: str, dest: Path) -> tuple[Path, str]:
    """Fetch a file with NO planning-time pin; record its sha (TOFU, named in plan)."""
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=HF_DATA_REPO, filename=repo_path, repo_type="dataset")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    actual = sha256_file(dest)
    log.info("record-only fetch: %s -> %s (sha256=%s)", repo_path, dest, actual[:12])
    return dest, actual


_MODEL_REPO_FILES: list[str] | None = None


def _model_repo_files() -> list[str]:
    """list_repo_files at the PINNED adapter revision (NOT snapshot_download —
    siblings-truncation gotcha on large repos)."""
    global _MODEL_REPO_FILES
    if _MODEL_REPO_FILES is None:
        from huggingface_hub import list_repo_files

        _MODEL_REPO_FILES = list(list_repo_files(HF_MODEL_REPO, revision=ADAPTER_REVISION))
    return _MODEL_REPO_FILES


def _fetch_adapter(source: str, adapters_root: Path) -> Path:
    """Download one frozen #411 adapter @ ADAPTER_REVISION into the snapshot layout."""
    from huggingface_hub import hf_hub_download

    sub = ADAPTER_PATH_TMPL.format(source=source)
    repo_files = [f for f in _model_repo_files() if f.startswith(f"{sub}/")]
    missing_remote = [f for f in REQUIRED_ADAPTER_FILES if f"{sub}/{f}" not in repo_files]
    if missing_remote:
        raise RuntimeError(
            f"Frozen adapter {sub} incomplete ON THE HUB @ {ADAPTER_REVISION}: "
            f"missing {missing_remote} (found {len(repo_files)} files under the prefix)."
        )
    adapter_dir = adapters_root / "_snapshot" / sub
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for repo_path in repo_files:
        cached = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename=repo_path, revision=ADAPTER_REVISION
        )
        shutil.copyfile(cached, adapter_dir / Path(repo_path).name)
    missing_local = [f for f in REQUIRED_ADAPTER_FILES if not (adapter_dir / f).exists()]
    if missing_local:
        raise RuntimeError(f"Frozen adapter {sub}: missing {missing_local} in {adapter_dir}")
    log.info("adapter OK: %s (%d files @ %s)", adapter_dir, len(repo_files), ADAPTER_REVISION[:12])
    return adapter_dir


def _assert_git_input(rel: str) -> Path:
    """Assert a frozen input that lives IN GIT exists at the repo/worktree root."""
    path = repo_root_from_module() / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen git input missing: {path}. Run "
            f"`git sparse-checkout add {Path(rel).parts[0]}/{Path(rel).parts[1]}` "
            f"(sparse worktrees exclude eval_results bulk) or `git pull`."
        )
    return path


def _assert_manifest_sha(data_path: Path, manifest_path: Path, label: str) -> str:
    """Assert a P0/implementation-time artifact matches its committed sha manifest."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"{label} missing: {data_path}. The producing VM phase must run + commit "
            f"BEFORE pod launch (plan §4 phase map)."
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{label} sha manifest missing: {manifest_path} (the producing phase "
            f"writes it next to the artifact; both are committed together)."
        )
    expected = json.loads(manifest_path.read_text())["sha256"]
    actual = sha256_file(data_path)
    if actual != expected:
        raise RuntimeError(
            f"{label} content drift: {data_path} sha256={actual} != committed "
            f"manifest {expected}. The artifact and its manifest are out of sync."
        )
    log.info("%s OK: %s (sha256=%s)", label, data_path, actual[:12])
    return actual


def prefetch(
    *,
    cells: list[tuple[str, str, int]],
    data_root: Path,
    adapters_root: Path,
    smoke_gate: bool = True,
) -> dict[str, str]:
    """Fetch + pin everything the requested cells need. Returns a manifest dict."""
    manifest: dict[str, str] = {}
    repo_root = repo_root_from_module()

    train_sources = sorted(
        {s for s, arm, _ in cells if arm in TRAIN_ARMS}, key=lambda s: SOURCES.index(s)
    )
    has_panel_build = any((s, a) == ("panel", "build") for s, a, _ in cells)
    has_base_pass = any((s, a) == ("base", "pass") for s, a, _ in cells)
    parity_sources = sorted(
        {s for s, arm, _ in cells if arm == "parity"},
        key=lambda s: PARITY_SOURCES.index(s),
    )
    has_prefix_arm = any(arm == "arm_prefix" for _, arm, _ in cells)
    # G1 fires after the smoke cell (villain:arm_onpolicy:42) -> needs the
    # frozen villain adapter + frozen eval_50 even without a parity cell.
    g1_in_scope = smoke_gate and ("villain", "arm_onpolicy", 42) in cells
    adapter_sources = sorted(set(parity_sources) | ({"villain"} if g1_in_scope else set()))

    # -- frozen claims (pinned) ------------------------------------------------
    eval50 = data_root / "wrong_claims" / "eval_50.jsonl"
    _fetch_pinned(f"{FROZEN_DATA_PREFIX}/data/wrong_claims/eval_50.jsonl", eval50)
    manifest["eval_50"] = str(eval50)
    train200 = data_root / "wrong_claims" / "train_200.jsonl"
    _fetch_pinned(f"{FROZEN_DATA_PREFIX}/data/wrong_claims/train_200.jsonl", train200)
    manifest["train_200"] = str(train200)

    # -- audited claims (P0 output, committed to git with a sha manifest) ------
    if train_sources or has_base_pass:
        eval60 = repo_root / "data" / "issue_612" / "wrong_claims" / "eval_60.jsonl"
        sha = _assert_manifest_sha(
            eval60, eval60.with_suffix(".jsonl.sha256.json"), "audited eval_60"
        )
        manifest["eval_60"] = str(eval60)
        manifest["eval_60_sha256"] = sha

    # -- frozen git inputs -----------------------------------------------------
    for rel in (
        FROZEN_JOIN_RELPATH,
        NEG_MEMBERSHIP_RELPATH,
        ANALYZE_SUMMARY_RELPATH,
        BASE_PANEL_RATES_RELPATH,
    ):
        manifest[Path(rel).name] = str(_assert_git_input(rel))

    # -- frozen #411 training pools (pinned) ------------------------------------
    for source in train_sources:
        dest = data_root / "pools_411" / f"{source}_seed42" / "train_pool.jsonl"
        _fetch_pinned(f"{FROZEN_DATA_PREFIX}/training_pools/{source}_seed42/train_pool.jsonl", dest)
        manifest[f"pool_{source}"] = str(dest)

    # -- #591 panel artifacts (record-only sha; no planning-time pin exists) ----
    if has_panel_build or train_sources:
        dest, sha = _fetch_record_only(
            f"{I591_DATA_PREFIX}/e2/twin_validation.json",
            data_root / "i591" / "twin_validation.json",
        )
        manifest["twin_validation"] = str(dest)
        manifest["twin_validation_sha256_record_only"] = sha

    # -- arm_prefix conversational-prefix questions (committed at impl time) ----
    if has_prefix_arm:
        pq = repo_root / "data" / "issue_612" / "prefix_questions.jsonl"
        sha = _assert_manifest_sha(pq, pq.with_suffix(".jsonl.sha256.json"), "prefix questions")
        manifest["prefix_questions"] = str(pq)
        manifest["prefix_questions_sha256"] = sha

    # -- frozen #411 adapters (parity / G1) -------------------------------------
    for source in adapter_sources:
        manifest[f"adapter_{source}"] = str(_fetch_adapter(source, adapters_root))

    manifest_path = data_root / "prefetch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("prefetch complete: %d entries -> %s", len(manifest), manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 prefetch — SHA256-pinned frozen-input fetch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cells", type=parse_cells, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_612"))
    parser.add_argument("--adapters-root", type=Path, default=Path("/workspace/adapters_411"))
    parser.add_argument(
        "--no-smoke-gate",
        action="store_true",
        help="Skip the G1 adapter fetch even when the smoke cell is in --cells.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=prefetch] %(message)s", stream=sys.stdout
    )
    prefetch(
        cells=args.cells,
        data_root=args.data_root,
        adapters_root=args.adapters_root,
        smoke_gate=not args.no_smoke_gate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
