"""Issue #621 artifact uploader (pod-side, runs BEFORE the results sentinel).

Pushes every upload-policy artifact class the pipeline produces to the HF
data repo, with fail-loud fresh-listing verification per class:

  1. Raw completions — via the canonical
     ``upload_raw_completions_to_data_repo`` helper (rglobs the per-cell
     ``raw_generations/<slug>/raw_completions.json`` files the eval
     emission mode writes).
  2. Training mixes (``training_mixes/*.jsonl``) →
     ``issue621_rank1_readwrite/training_mixes/``.
  3. Eval shift tensors (``eval/*__shift.{pt,json}`` — plan-referenced
     analysis inputs, #521 rule) →
     ``issue621_rank1_readwrite/analysis_tensors/shifts/``.
  4. Band trajectories (``cells/*/band_trajectory.json`` +
     ``cells/*/marker_band_stop_result.json``) →
     ``issue621_rank1_readwrite/trajectories/``.
  5. Eval emission JSONs (``eval/*__emission.json``) →
     ``issue621_rank1_readwrite/eval/`` (round-2, concern
     ``analysis-fetch-missing-shifts``: the GCP instance self-deletes, so
     the HF copies are the ONLY durable eval JSONs the off-pod Phase A
     can fetch).
  6. Train-side cell metadata (``anchor_smoke/*.json`` + ``sweep/*.json``)
     → ``issue621_rank1_readwrite/train_meta/{anchor_smoke,sweep}/``
     (round-2: ``issue621_analyze.py`` joins band_stop_fired /
     band_stop_step / final_source_delta_nats from these for the
     banded-vs-below-band splits — they must survive instance deletion).

Each class is ONE ``upload_folder`` commit (HF 256 commits/hr rule) with a
bounded 5xx retry, then verified against a FRESH complete listing
(``list_repo_files_complete`` — raw ``list_repo_files`` silently truncates
on big repos) — any expected path missing raises (the pipeline's ``set -e``
aborts before ``[phase=done]``).

CLI:
    uv run python scripts/i621_upload_artifacts.py [--out-root eval_results/issue_621]
"""

# math/scientific notation in docstrings

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from explore_persona_space.experiments.issue_621 import (
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_BUCKET,
    HF_DATA_REPO,
    HF_TRAIN_MIX_PATH_PREFIX,
)

log = logging.getLogger("issue_621.upload")

_RETRIES = 3
_BACKOFFS = (30, 60, 120)


def _upload_folder_with_retry(api, **kwargs) -> None:
    """One upload_folder commit with bounded retry on transient 5xx errors.

    4xx (auth/quota/permission) failures re-raise immediately — retrying a
    403 quota gate changes nothing (upload-policy rule).
    """
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(_RETRIES):
        try:
            api.upload_folder(**kwargs)
            return
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            transient = status is not None and 500 <= status < 600
            if not transient or attempt == _RETRIES - 1:
                raise
            wait = _BACKOFFS[attempt]
            log.warning("upload_folder 5xx (%s); retry %d/%d in %ds", status, attempt + 1, 3, wait)
            time.sleep(wait)


def _verify_on_hub(api, expected_paths: list[str], label: str) -> None:
    """FRESH listing verification — every expected path must be on the Hub.

    Uses ``list_repo_files_complete`` (paginated tree API): raw
    ``list_repo_files`` reads ``siblings`` and silently truncates at ~7.9k
    entries — the data repo is well past that.
    """
    from explore_persona_space.orchestrate.hub import list_repo_files_complete

    listed = set(list_repo_files_complete(api, HF_DATA_REPO, repo_type="dataset"))
    missing = [p for p in expected_paths if p not in listed]
    if missing:
        raise RuntimeError(
            f"{label}: upload verification FAILED — {len(missing)} path(s) "
            f"missing on Hub, first: {missing[:3]}"
        )
    log.info("%s: %d file(s) verified on Hub.", label, len(expected_paths))


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="eval_results/issue_621")
    ap.add_argument(
        "--skip-raw-completions",
        action="store_true",
        help="Skip class 1 (smoke runs where no emission eval ran).",
    )
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    api = HfApi()

    # 1. Raw completions (canonical helper; fail-loud inside).
    if args.skip_raw_completions:
        log.warning("Skipping raw-completions upload (--skip-raw-completions).")
    else:
        raw_files = sorted(out_root.rglob("raw_completions.json"))
        if not raw_files:
            raise RuntimeError(
                f"no raw_completions.json under {out_root} — the emission eval "
                "did not run / wrote elsewhere. Refusing to advance (upload "
                "policy: raw completions land before termination)."
            )
        log.info("Uploading %d raw_completions.json via canonical helper", len(raw_files))
        upload_raw_completions_to_data_repo(
            experiment_name=HF_BUCKET,
            eval_results_dir=out_root,
        )

    # 2. Training mixes.
    mixes = sorted((out_root / "training_mixes").glob("*.jsonl"))
    if not mixes:
        raise RuntimeError(f"no training mixes under {out_root}/training_mixes")
    _upload_folder_with_retry(
        api,
        folder_path=str(out_root / "training_mixes"),
        path_in_repo=HF_TRAIN_MIX_PATH_PREFIX,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
        commit_message="task #621 training mixes",
    )
    _verify_on_hub(
        api,
        [f"{HF_TRAIN_MIX_PATH_PREFIX}/{m.name}" for m in mixes],
        "training mixes",
    )

    # 3. Eval shift tensors + slot-stats JSONs (plan-referenced analysis
    # inputs, #521 rule). BOTH suffixes verified — the off-pod analyzer's
    # load_eval_cells raises without the JSONs (concern
    # analysis-fetch-missing-shifts), so a pt-only verification is not
    # enough.
    shifts = sorted((out_root / "eval").glob("*__shift.pt")) + sorted(
        (out_root / "eval").glob("*__shift.json")
    )
    if not shifts:
        raise RuntimeError(f"no shift tensors under {out_root}/eval — eval did not run?")
    _upload_folder_with_retry(
        api,
        folder_path=str(out_root / "eval"),
        path_in_repo=f"{HF_ANALYSIS_TENSORS_PREFIX}/shifts",
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*__shift.pt", "*__shift.json"],
        commit_message="task #621 eval shift tensors + slot stats",
    )
    _verify_on_hub(
        api,
        [f"{HF_ANALYSIS_TENSORS_PREFIX}/shifts/{s.name}" for s in shifts],
        "shift tensors + slot-stats JSONs",
    )

    # 4. Band trajectories (per-cell JSON, tiny).
    traj_files = sorted((out_root / "cells").glob("*/band_trajectory.json"))
    if not traj_files:
        raise RuntimeError(f"no band_trajectory.json under {out_root}/cells/*/")
    _upload_folder_with_retry(
        api,
        folder_path=str(out_root / "cells"),
        path_in_repo=f"{HF_BUCKET}/trajectories",
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*/band_trajectory.json", "*/marker_band_stop_result.json"],
        commit_message="task #621 band-stop trajectories",
    )
    _verify_on_hub(
        api,
        [f"{HF_BUCKET}/trajectories/{t.parent.name}/band_trajectory.json" for t in traj_files],
        "band trajectories",
    )

    # 5. Eval emission JSONs (round-2, concern analysis-fetch-missing-shifts:
    # eval_results/ is never committed to git pod-side, so the HF copy is the
    # only durable one once the instance self-deletes).
    emissions = sorted((out_root / "eval").glob("*__emission.json"))
    if emissions:
        _upload_folder_with_retry(
            api,
            folder_path=str(out_root / "eval"),
            path_in_repo=f"{HF_BUCKET}/eval",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            allow_patterns=["*__emission.json"],
            commit_message="task #621 eval emission JSONs",
        )
        _verify_on_hub(
            api,
            [f"{HF_BUCKET}/eval/{e.name}" for e in emissions],
            "eval emission JSONs",
        )
    elif args.skip_raw_completions:
        log.warning("no emission JSONs (smoke run, --skip-raw-completions) — class 5 skipped.")
    else:
        raise RuntimeError(
            f"no *__emission.json under {out_root}/eval — emission eval did not "
            "run; refusing to advance."
        )

    # 6. Train-side cell metadata (anchor_smoke/ + sweep/ per-cell JSONs) —
    # the analyzer's banded-vs-below-band join inputs (round-2).
    n_meta = 0
    for sub in ("anchor_smoke", "sweep"):
        d = out_root / sub
        metas = sorted(d.glob("*.json")) if d.is_dir() else []
        if not metas:
            continue
        n_meta += len(metas)
        _upload_folder_with_retry(
            api,
            folder_path=str(d),
            path_in_repo=f"{HF_BUCKET}/train_meta/{sub}",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            allow_patterns=["*.json"],
            commit_message=f"task #621 train-side cell metadata ({sub})",
        )
        _verify_on_hub(
            api,
            [f"{HF_BUCKET}/train_meta/{sub}/{m.name}" for m in metas],
            f"train meta ({sub})",
        )
    if n_meta == 0:
        raise RuntimeError(
            f"no train-side cell JSONs under {out_root}/{{anchor_smoke,sweep}} — "
            "the analyzer's banded/below-band split is unrunnable off-pod without them."
        )

    log.info("ALL artifact classes uploaded + verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
