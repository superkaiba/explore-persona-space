#!/usr/bin/env python3
"""Exact-fidelity #2552 judge-wave adapter over the reviewed parent instruments.

This follow-up evaluates the fresh LMSYS-only assistant-turn SAE against the public
layer-19 per-token comparator under max and sum pooling.  It deliberately reuses the
already-reviewed #2552 Batch-API implementation at its result commit instead of
forking the judge prompts, parsers, retry policy, completeness gates, or raw-output
persistence.  The only protocol adaptation is the registered three configurations:

    rep_ta, pt_max, pt_sum

The matched context-vector SAE is an additional training arm, not a paper-replication
configuration, and therefore does not enter these headline paper targets.

Required prepared inputs under ``<out-root>`` (written by the exactrep eval-input
driver) are:

  judge_aggregates/g2_decision.json
  eval_texts.jsonl
  inputs/<hf-prefix>/analysis_tensors/eval_lists/feature_lists_2000turns.json
  inputs/<hf-prefix>/analysis_tensors/eval_lists/<config shards>
  inputs/<hf-prefix>/raw_completions/mining/top25_{rep_ta,pt}*.jsonl

The parent source must be checked out at ``PARENT_COMMIT``.  By default the adapter
uses the repository's issue-2552 worktree; ``--parent-worktree`` can point at any
checkout of that exact commit.  This pin makes the dependency reproducible and fails
loud rather than silently drifting with a branch tip.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

PARENT_COMMIT = "427af35b8c2d5d5d5f40de72298b0acb234fe7e0"
DEFAULT_PARENT_WORKTREE = PROJECT_ROOT / ".claude" / "worktrees" / "issue-2552"
CONFIGS = ("rep_ta", "pt_max", "pt_sum")
CONFIG_FAMILY = {"rep_ta": "rep_ta", "pt_max": "pt", "pt_sum": "pt"}
W6_LABELS = tuple("ABC")
RUN_PHASES = ("pilot-w1", "w1", "w2", "pilot-w4", "w4", "pilot-w5", "w5", "w6")


def _checkout_sha(worktree: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip()


def load_parent(worktree: Path) -> ModuleType:
    """Load and patch the pinned parent judge module for the exactrep config set."""
    worktree = worktree.resolve()
    source = worktree / "scripts" / "issue2552_judge_waves.py"
    assert source.exists(), f"parent judge source missing: {source}"
    got = _checkout_sha(worktree)
    assert got == PARENT_COMMIT, f"parent judge checkout drift: {got} != {PARENT_COMMIT}"
    name = "issue2552_judge_waves_exactrep_parent"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    module.TA_FAMILIES = ("rep_ta",)
    module.ALL_FAMILIES = ("rep_ta", "pt")
    module.CONFIGS = CONFIGS
    module.CONFIG_FAMILY = dict(CONFIG_FAMILY)
    module.W5_PAIRS = tuple(itertools.combinations(CONFIGS, 2))
    module.W6_LABELS = W6_LABELS
    module.W6_SYSTEM = (
        "You will be shown a structured summary of one assistant conversation turn, followed"
        " by three lists of feature descriptions labeled A through C, each derived from that"
        " same turn by a different method. Rank ALL THREE lists from best to worst coverage"
        " of what the summary says about the turn. Reason briefly first, then answer. Output"
        " ONLY a single JSON object and nothing else. Use the form "
        '{"reason": "<brief reasoning>", "ranking": ["<best label>", ..., "<worst label>"]}'
        " listing all three labels exactly once."
    )
    module.WAVE_SYSTEMS["w6"] = module.W6_SYSTEM

    def exact_paths(args) -> SimpleNamespace:
        work = Path(args.out_root)
        if args.smoke:
            work = work / "smoke"
        agg = work / "judge_aggregates"
        dere = work / "dere_repl"
        for d in (work, agg, dere):
            d.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(work=work, agg=agg, dere=dere, inputs=work / "inputs")

    module._paths = exact_paths
    return module


def _args_for_parent(args) -> SimpleNamespace:
    return SimpleNamespace(
        out_root=args.out_root,
        hf_prefix=args.hf_prefix,
        dry_run=args.dry_run,
        smoke=args.smoke,
        skip_upload=args.skip_upload,
        allow_below_floor=args.allow_below_floor,
        max_chunks=0,
    )


def _run_all(parent: ModuleType, args) -> None:
    for phase in RUN_PHASES:
        print(f"[exactrep-all] phase {phase}", flush=True)
        parent.PHASES[phase](args)
    if not (args.dry_run or args.smoke):
        parent._t2552().C.write_json_atomic(
            Path(args.out_root) / "w_all_done.json",
            {
                "status": "done",
                "waves": ["w1", "w2", "w4", "w5", "w6"],
                "configs": list(CONFIGS),
                "parent_commit": PARENT_COMMIT,
            },
        )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wave", choices=[*RUN_PHASES, "all"])
    ap.add_argument("--out-root", type=Path, default=Path("data/issue_2552/exactrep_judge"))
    ap.add_argument("--hf-prefix", default="issue2552_derreplication/exactrep")
    ap.add_argument("--parent-worktree", type=Path, default=DEFAULT_PARENT_WORKTREE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--allow-below-floor", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps([*RUN_PHASES, "all"]))
        return 0
    parent = load_parent(args.parent_worktree)
    if args.import_check:
        assert parent.CONFIGS == CONFIGS
        assert parent.W6_LABELS == W6_LABELS
        print(f"[import-check] parent={PARENT_COMMIT[:12]} configs={CONFIGS}")
        return 0
    assert args.wave, "--wave is required"
    pargs = _args_for_parent(args)
    if args.wave == "all":
        _run_all(parent, pargs)
    else:
        parent.PHASES[args.wave](pargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
