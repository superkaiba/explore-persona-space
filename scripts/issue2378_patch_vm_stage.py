#!/usr/bin/env python3
"""Issue #2378 causal-patching-arms — VM-side stage-back of the pod harvest.

Downloads a round's HF prefixes into the judge/analysis --patch-root layout
(anchors/grid/confirm/bank/meta rollout JSONLs; ``--tensors`` adds the
``analysis_tensors`` ``<stage>_va`` npz dirs as ``<stage>/va/``). The
orchestrator-owned stage-back step the r18 review accepted as m9: both VM
legs fail loud on absence, this script is what makes them present. Small
text-only pull by default (~306 files, tens of MB) — VM staging is in-policy.

Round-scoped legs (dana-behavior-confirm): ``--hf-suffix _danaconf`` selects
the round's own prefixes, ``--dest`` its round root, ``--stages`` filters to
the stage dirs a leg actually consumes (e.g. ``anchors`` + ``--tensors`` from
the ORIGINAL prefixes for anchor floors/ceilings; ``confirm,bank,meta`` +
``--tensors`` from the round prefixes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2378_common as cm  # noqa: E402


def _dest_rel(prefix: str, path: str, stages: set[str] | None) -> str | None:
    """Map one repo file path under ``prefix`` to its --patch-root-relative
    destination, or None when ``--stages`` filters it out.

    Raw prefix: top-level ``<stage>/<name>.jsonl`` (anchors/grid/confirm) maps
    to ``<stage>/rollouts/<name>`` — the pod uploaded each stage's rollouts/
    CONTENT directly under the stage name while the judge/analysis globs
    expect ``<stage>/rollouts/*.jsonl``; everything else keeps its relative
    path. Tensor prefix: ``<stage>_va/<name>`` maps to ``<stage>/va/<name>``
    (the pod uploads ``<stage>/va`` as ``<stage>_va``)."""
    rel = path[len(prefix) + 1 :]
    stage, _, name = rel.partition("/")
    base = stage[: -len("_va")] if stage.endswith("_va") else stage
    if stages is not None and base not in stages:
        return None
    if stage.endswith("_va"):
        return f"{base}/va/{name}"
    if stage in ("anchors", "grid", "confirm") and name.endswith(".jsonl") and "/" not in name:
        return f"{stage}/rollouts/{name}"
    return rel


def stage_back(
    dest: Path, hf_suffix: str = "", stages: set[str] | None = None, tensors: bool = False
) -> int:
    """Stage the round's HF prefixes into ``dest`` (idempotent: present
    targets skip). Fails loud on an empty listing or an all-filtered
    selection (empty selection is never a silent no-op)."""
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    prefixes = [f"{cm.HF_PREFIX}/raw_completions/causal_patching{hf_suffix}"]
    if tensors:
        prefixes.append(f"{cm.HF_PREFIX}/analysis_tensors/causal_patching{hf_suffix}")
    n = tot = 0
    for prefix in prefixes:
        files = hub.list_hf_files_under_path(api, cm.HF_DATA_REPO, prefix, repo_type="dataset")
        if not files:
            raise RuntimeError(f"empty stage-back listing under {prefix} (fail loud)")
        for path in sorted(files):
            rel = _dest_rel(prefix, path, stages)
            if rel is None:
                continue
            tot += 1
            target = dest / rel
            if target.exists():
                continue
            got = hub.retry_transient(
                lambda p=path: hf_hub_download(
                    cm.HF_DATA_REPO, p, repo_type="dataset", local_dir="/tmp/i2378_stageback"
                ),
                what=f"download {path}",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(got).read_bytes())
            n += 1
    if tot == 0:
        raise RuntimeError(
            f"--stages {sorted(stages or set())} filtered every file under {prefixes} "
            "(empty selection — fail loud)"
        )
    print(f"[stage-back] {n} downloaded, {tot - n} already present -> {dest}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dest", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "patch_round"))
    ap.add_argument("--hf-suffix", default="", help="round prefix suffix (e.g. _danaconf)")
    ap.add_argument(
        "--stages",
        default=None,
        help="comma filter over stage dirs (bank,anchors,grid,confirm,meta,judge_persona); "
        "default: everything under the prefix",
    )
    ap.add_argument(
        "--tensors",
        action="store_true",
        help="also stage analysis_tensors <stage>_va/ -> <stage>/va/ (same --stages filter)",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    stages = (
        {s.strip() for s in args.stages.split(",") if s.strip()}
        if args.stages is not None
        else None
    )
    if args.stages is not None and not stages:
        raise SystemExit("--stages given but empty after parsing (fail loud)")
    return stage_back(
        Path(args.dest), hf_suffix=args.hf_suffix, stages=stages, tensors=args.tensors
    )


if __name__ == "__main__":
    sys.exit(main())
