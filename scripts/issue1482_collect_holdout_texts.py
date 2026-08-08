"""Issue #1482 context-extremes round — collect #1738 multi-turn holdout texts.

Reuses ``issue1738_characterize._collect_holdout_texts`` verbatim (per-chunk HF
download -> scan -> unlink, checkpointed to a gitignored scratch; text never
logged) to build the ci -> {last_user, history_tail, response, corpus} cache the
context-extremes dashboard (D1), the blinded qualitative read (D2), and nothing
else consumes. The needed set is every ci in the banked three-arm per-context
summary (eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv),
i.e. the full 9,941-row holdout. 0 GPU; CPU + HF download only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import issue1738_characterize as CH  # noqa: E402

CSV = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1738"
    / "bare_query"
    / ("percontext_summary_L19_ridge.csv")
)
SCRATCH = PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-prefix", default="issue1738_multiturn")
    ap.add_argument("--local-raw-dir", default=None)
    ap.add_argument("--scratch", default=str(SCRATCH))
    args = ap.parse_args()

    with open(CSV, encoding="utf-8") as f:
        needed = {int(r["ci"]) for r in csv.DictReader(f)}
    print(f"[collect] needed cis: {len(needed)}", flush=True)
    found = CH._collect_holdout_texts(args, needed)
    missing = sorted(needed - set(found))
    print(f"[collect] found {len(found)}/{len(needed)}; missing: {missing[:20]}", flush=True)
    print(f"[collect] cache: {Path(args.scratch) / 'judge_texts.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
