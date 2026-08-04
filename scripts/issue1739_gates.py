"""Phase-0 gate CLI for issue #1739 (round A).

Runs the executable gates (Gate 0 store sha-pin probe, r_B bank probe, Gate 3
staged-layout probe) and the argument-validated round-B stubs (Gates 1-2).
Writes a JSON gate report with reproducibility metadata.

Usage:
    uv run python scripts/issue1739_gates.py --gate all
    uv run python scripts/issue1739_gates.py --gate 0 --store-revision e5901706
"""

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps + credentials bind BEFORE any heavy import (shared-VM rule, #847).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.experiments.issue_1739 import constants, gates  # noqa: E402


def _git_commit() -> str:
    """Best-effort git commit for reproducibility metadata."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> int:
    """Run the requested Phase-0 gates and write the gate report JSON."""
    parser = argparse.ArgumentParser(description="Issue #1739 Phase-0 gates (round A)")
    parser.add_argument(
        "--gate",
        choices=["0", "rb", "3", "1", "2", "all"],
        default="all",
        help="which gate to run; 'all' = the executable set {0, rb, 3} (1-2 are round-B stubs)",
    )
    parser.add_argument("--store-revision", default=constants.STORE_REVISION)
    parser.add_argument("--rb-revision", default=constants.RB_REVISION)
    parser.add_argument(
        "--local-dir",
        default="data/issue_1739/hf_dl/gate_probe",
        help="staging dir for the Gate-3 1-file probe downloads",
    )
    parser.add_argument(
        "--report-path",
        default="eval_results/issue_1739/gates/phase0_gate_report.json",
        help="gate report JSON output path (use a scratch path for smokes)",
    )
    parser.add_argument("--behavior", default=None, help="behavior for gates 1-2")
    parser.add_argument(
        "--dv-json",
        default=None,
        help="dv_build labeling.json path (gates 1-2 report mode over existing DV rows)",
    )
    parser.add_argument(
        "--run-pilot",
        action="store_true",
        help="gate 1 full pilot driver (staging -> generation -> judging; GPU+API — round C)",
    )
    parser.add_argument("--n-pilot", type=int, default=None)
    parser.add_argument("--staged-dir", default="data/issue_1739/staged")
    parser.add_argument("--out-root", default="eval_results/issue_1739")
    parser.add_argument("--stream-cap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def _dv_rows() -> list[dict]:
        if not args.dv_json:
            parser.error("--dv-json is required for gate 1/2 report mode")
        return json.loads(Path(args.dv_json).read_text())["rows"]

    reports: dict[str, dict] = {}
    if args.gate in ("0", "all"):
        reports["gate0"] = gates.gate0_store_pin_probe(revision=args.store_revision)
    if args.gate in ("rb", "all"):
        reports["rb_probe"] = gates.rb_bank_probe(revision=args.rb_revision)
    if args.gate in ("3", "all"):
        reports["gate3"] = gates.gate3_staged_layout_probe(
            Path(args.local_dir), revision=args.store_revision
        )
    if args.gate == "1":
        if not args.behavior:
            parser.error("--behavior is required for gate 1")
        if args.run_pilot:
            pilot_kwargs = {} if args.n_pilot is None else {"n_pilot": args.n_pilot}
            reports["gate1_pilot"] = gates.run_gate1_pilot(
                args.behavior,
                out_root=args.out_root,
                staged_dir=args.staged_dir,
                seed=args.seed,
                stream_cap=args.stream_cap,
                **pilot_kwargs,
            )
        else:
            report_kwargs = {} if args.n_pilot is None else {"n_pilot": args.n_pilot}
            reports["gate1"] = gates.gate1_yield_report(
                _dv_rows(), behavior=args.behavior, **report_kwargs
            )
    if args.gate == "2":
        if not args.behavior:
            parser.error("--behavior is required for gate 2")
        reports["gate2"] = gates.gate2_spread_floor(_dv_rows(), behavior=args.behavior)

    payload = {
        "issue": 1739,
        "phase": "phase0_gates",
        "gate_arg": args.gate,
        "reports": reports,
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
