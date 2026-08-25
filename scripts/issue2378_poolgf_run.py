"""Issue #2378 — corrective pool_gf + LOFO pod run (interpretation-critique r1 fix).

Thin pod-side sequencer over REVIEWED code: stages the full activation store
(fits-d shape: ALL cells' L* npz) + the sibling percell rowstats sidecars via
the reviewed dispatch helpers, then runs scripts/issue2378_pool.py under
--global-family-folds (pool -> h5 -> lofo; science code reviewed at r15,
commit f69444a313), harvests eval_results/issue_2378/pool_gf/ to the issue
branch, and writes the completion sentinel. Orchestration glue only — no new
science code; every phase emits its [phase=...] breadcrumb and the terminal
[phase=done] per the pod-side reporting contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import issue2378_common as cm
import issue2378_dispatch as d


def _child(argv: list[str]) -> int:
    """Run a pool.py phase as a foreground child; stream output; return rc."""
    d._log(f"[poolgf] exec: {' '.join(argv)}")
    return subprocess.run(argv).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-root", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "p6_stage"))
    ap.add_argument("--ledger-root", default=str(cm.LEDGER_ROOT))
    ap.add_argument("--no-jitter", action="store_true", default=True)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    d._phase_line("poolgf_stage")
    ledger_root = Path(args.ledger_root)
    d.assert_headroom("poolgf", Path(args.stage_root))
    d._git_pull_rebase()
    g2b = json.loads((ledger_root / "g2b_report.json").read_text(encoding="utf-8"))
    survivors = [c for c in cm.ALL_CELLS if c in g2b["survivors"]]
    surv_arg = ",".join(survivors)
    lstar = d.resolve_lstar(ledger_root)
    store_root = d.stage_p6(args, "fits-d", lstar)
    d._stage_sidecars(ledger_root)

    py = sys.executable
    pool = str(cm.REPO_ROOT / "scripts" / "issue2378_pool.py")
    store = ["--store-root", str(store_root), "--ledger-root", str(ledger_root)]
    for phase, extra in (
        ("pool", ["--cells", surv_arg]),
        ("h5", ["--cells", surv_arg]),
        ("lofo", []),
    ):
        d._phase_line(f"poolgf_{phase}")
        rc = _child([py, pool, "--phase", phase, "--global-family-folds", *store, *extra])
        if rc != 0:
            d._log(f"[poolgf] phase {phase} rc={rc} — failing loud (no sentinel)")
            return rc

    d._phase_line("poolgf_harvest")
    d.git_harvest(
        [
            "eval_results/issue_2378/pool_gf/*.json",
            "eval_results/issue_2378/pool_gf/own_ceilings/*.json",
            "eval_results/issue_2378/pool_gf/lofo/*.json",
        ],
        f"task #{d.ISSUE}: P6 pool_gf + LOFO (global family folds; interp-r1 fix)",
    )
    d.write_sentinel(
        args,
        "epm:progress",
        {
            "phase": "poolgf complete (pool+h5+lofo under --global-family-folds)",
            "survivors": survivors,
            "eval_json_paths": ["eval_results/issue_2378/pool_gf/"],
            "note": "corrective re-run for interp-critique r1 (family-exposed pooled folds)",
        },
        gate=None,
        blocks_pipeline=False,
    )
    d._phase_line("done")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    sys.exit(main())
