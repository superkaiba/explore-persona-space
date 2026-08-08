"""Stage issue2162 P7 inputs from HF via the canonical scoped-prefix helper (#1402).

Run from /workspace/explore-persona-space with .env already sourced.
Fail-loud: any count mismatch or per-file failure aborts before analysis runs.
"""

import sys
from pathlib import Path

from explore_persona_space.orchestrate.hub import stage_hub_prefix

REPO = "superkaiba1/explore-persona-space-data"
STAGE = Path("/workspace/issue2162_stage")

PREFIXES = [
    ("issue2162_ctxinfo/raw_completions/grid", 234),
    ("issue2162_ctxinfo/raw_completions/anchors", 16),
    ("issue2162_ctxinfo/analysis_tensors/va_store", 234),
    ("issue2162_ctxinfo/analysis_tensors/vc_bank", 4),
    ("issue2162_ctxinfo/raw_completions/judge_raw", 748),
]


def main() -> None:
    for prefix, expected in PREFIXES:
        files = stage_hub_prefix(REPO, prefix, STAGE)
        n = len(files)
        print(f"STAGED {prefix}: {n} files (expected {expected})", flush=True)
        if n != expected:
            print(f"COUNT MISMATCH for {prefix}: got {n} expected {expected}", flush=True)
            sys.exit(3)
    print("STAGING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
