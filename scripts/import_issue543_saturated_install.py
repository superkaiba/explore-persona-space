"""Import the #543 saturated phase1-FINAL installs as the #570 ``saturated`` variant.

Follow-up round ``saturated-install-em-eraser`` Step 0 (plan v4 §4): per seed,
download ``adapters/issue543/r50_seed<S>_phase1`` from the Hub at the pinned
model-repo revision, assert the adapter files + gauge, and write the
``phase1_result.json`` provenance record ``run_phase2`` reads at
``eval_results/issue_570/phase1_saturated/seed<S>/phase1_result.json``
(``run_issue543_ratio.py`` lines 1254-1267: ``install_excluded`` gate +
``final_adapter_path``). Honest provenance record, not a stub hack — the
record carries the Hub source + revision and the #543 stop-record values.

CPU-only; idempotent (re-download is a cache hit, the record is rewritten).

Usage:
    uv run python scripts/import_issue543_saturated_install.py            # all 3 seeds
    uv run python scripts/import_issue543_saturated_install.py --seeds 42 # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

log = bootstrap(log_name="import_issue543_saturated_install")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    EVAL_RESULTS_DIR,
    HUB_MODEL_REPO,
    cell_dir_570,
    repro_metadata,
)

ARM = "r50"
INSTALL_VARIANT = "saturated"
DEFAULT_SEEDS = (42, 137, 256)
# Model-repo `main` pinned at planning time (plan v4 §2/§10: the round's ONE
# manipulated variable is this start adapter; Hub-verified in the plan).
HUB_MODEL_REPO_REVISION_543_SATURATED = "0718c53058475cb8ee38c8f4802220cdde548672"
# The chain's install recipe of record trains attention projections only.
ALLOWED_TARGET_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
# #543 stop-record values copied into the provenance record (numeric only).
STOP_RECORD_KEYS = (
    "stop_step",
    "stop_reason",
    "last_trained_logp_mean",
    "last_argmax_rate",
    "stop_delta_mean_nats",
    "base_logp_mean",
)


def _download_adapter_root_files(sub: str, revision: str) -> Path:
    """Download the TOP-LEVEL files of a Hub adapter subfolder; fail loud.

    NOT snapshot_download(allow_patterns=...): on this repo it SILENTLY
    returns 0 files (siblings truncation — see hub.download_repo_subfolder
    docstring + the 2026-06-10 Stage-A smoke crash). NOT the recursive
    hub.download_repo_subfolder either: the #543 phase-1 subfolders carry
    their ladder checkpoint-*/ subdirs (~1.0 GB/seed of optimizer states the
    eraser never reads — measured in this round's smoke); ``recursive=False``
    fetches only the ~10 root adapter/tokenizer files (~40 MB). Destination
    mirrors the helper's deterministic layout so recorded paths are stable
    per pod.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from huggingface_hub.hf_api import RepoFile

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        matched = sorted(
            e.path
            for e in api.list_repo_tree(
                repo_id=HUB_MODEL_REPO, path_in_repo=sub, revision=revision, recursive=False
            )
            if isinstance(e, RepoFile)
        )
    except EntryNotFoundError as e:
        raise FileNotFoundError(
            f"Subfolder {sub}/ not found in {HUB_MODEL_REPO} @ {revision}"
        ) from e
    if not matched:
        raise FileNotFoundError(
            f"Subfolder {sub}/ in {HUB_MODEL_REPO} @ {revision} lists 0 root files"
        )
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    local_dir = (
        cache_root
        / "eps-subfolder-downloads"
        / f"models--{HUB_MODEL_REPO.replace('/', '--')}"
        / revision
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    for fname in matched:
        hf_hub_download(
            repo_id=HUB_MODEL_REPO,
            filename=fname,
            revision=revision,
            token=token,
            local_dir=str(local_dir),
        )
    missing = [f for f in matched if not (local_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(matched)} root files under {sub}/ failed to "
            f"materialize in {local_dir}: {missing[:3]}"
        )
    log.info("Downloaded %d root files for %s/%s @ %s", len(matched), HUB_MODEL_REPO, sub, revision)
    return local_dir / sub


def import_one(seed: int) -> dict:
    """Download + verify one seed's #543 phase1-FINAL adapter; write the record."""
    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

    sub = f"adapters/issue543/{ARM}_seed{seed}_phase1"
    adapter_dir = _download_adapter_root_files(sub, HUB_MODEL_REPO_REVISION_543_SATURATED)
    missing = [f for f in REQUIRED_FILES if not (adapter_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{sub} @ {HUB_MODEL_REPO_REVISION_543_SATURATED}: missing {missing} in {adapter_dir}"
        )
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    # Gauge assert (marker-leakage rule): no lm_head/embed_tokens targets,
    # modules_to_save empty — the logit DV is invalid otherwise.
    assert_gauge_free_adapter_config(cfg, context=str(adapter_dir))
    targets = {str(t) for t in (cfg.get("target_modules") or [])}
    if not targets or not targets.issubset(ALLOWED_TARGET_MODULES):
        raise AssertionError(
            f"{sub}: target_modules {sorted(targets)} is not a non-empty subset of "
            f"{sorted(ALLOWED_TARGET_MODULES)} — not the chain's install recipe."
        )

    # #543 stop-record provenance (committed in git on this branch).
    parent_path = EVAL_RESULTS_DIR / ARM / f"seed{seed}" / "phase1_result.json"
    parent = json.loads(parent_path.read_text())
    stop = {k: (parent.get("stop_record") or {}).get(k) for k in STOP_RECORD_KEYS}
    if stop["stop_step"] is None or stop["last_trained_logp_mean"] is None:
        raise RuntimeError(
            f"{parent_path}: stop_record missing stop_step / last_trained_logp_mean — "
            "wrong or truncated #543 parent record; refusing to write provenance."
        )

    cell = cell_dir_570(seed, "phase1", INSTALL_VARIANT)
    cell.mkdir(parents=True, exist_ok=True)
    record = {
        **repro_metadata(),
        "phase": "phase1",
        "arm": ARM,
        "seed": seed,
        "install_variant": INSTALL_VARIANT,
        # ── run_phase2 contract reads (run_issue543_ratio.py:1254-1267) ─────
        "install_excluded": False,
        "final_adapter_path": str(adapter_dir.resolve()),
        # ── provenance (plan v4 §4 Step 0) ──────────────────────────────────
        "import_source": {
            "hub_repo": HUB_MODEL_REPO,
            "hub_subfolder": sub,
            "hub_revision": HUB_MODEL_REPO_REVISION_543_SATURATED,
            "issue543_phase1_result_path": str(parent_path),
            "issue543_git_commit": parent.get("git_commit"),
            "issue543_stop_record": stop,
        },
    }
    out = cell / "phase1_result.json"
    out.write_text(json.dumps(record, indent=2))
    log.info(
        "seed %d: %s @ %s -> %s (stop_step=%s trained_logp_mean=%.4f argmax=%.4f) record=%s",
        seed,
        sub,
        HUB_MODEL_REPO_REVISION_543_SATURATED[:8],
        adapter_dir,
        stop["stop_step"],
        stop["last_trained_logp_mean"],
        stop["last_argmax_rate"],
        out,
    )
    return record


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Import #543 saturated phase1-FINAL installs as the #570 "
        "'saturated' install variant (provenance records + local adapter dirs).",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds to import (default: all three).",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if not seeds:
        raise SystemExit("--seeds parsed to an empty list.")
    digest = {}
    for seed in seeds:
        rec = import_one(seed)
        digest[f"seed{seed}"] = {
            "final_adapter_path": rec["final_adapter_path"],
            "stop_step": rec["import_source"]["issue543_stop_record"]["stop_step"],
            "last_trained_logp_mean": rec["import_source"]["issue543_stop_record"][
                "last_trained_logp_mean"
            ],
            "record": str(cell_dir_570(seed, "phase1", INSTALL_VARIANT) / "phase1_result.json"),
        }
    print(json.dumps(digest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
