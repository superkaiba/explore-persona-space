#!/usr/bin/env python3
"""#1775 P6 (CONDITIONAL on #1774): noise-ceiling-relative gain reporting.

Consumes the sibling task's per-direction decode-noise floor via
``--noise-floor-json <path>`` (NO hardcoded path — the artifact's location
binds at fold-in, plan section 4 P6) and re-expresses the per-arm ladder
gains relative to the floor. Never regenerates draws. Declared fallback: if
the sibling artifact has not landed, the clean-result ships RAW gains and a
same-issue follow-up round runs this script when it lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from issue1775_common import atomic_write_json, eval_dir, result_meta  # noqa: E402


def _extract_floor(d: dict) -> float:
    """Pull a scalar noise-floor R2-equivalent from the #1774 artifact.

    The exact schema binds at fold-in; accept the obvious spellings and fail
    loud otherwise (never a silent default)."""
    for k in ("noise_floor_r2", "decode_noise_floor", "floor_r2", "r2_ceiling", "ceiling_r2"):
        if isinstance(d.get(k), int | float):
            return float(d[k])
    raise SystemExit(
        f"noise-floor JSON carries none of the recognized scalar keys; top-level keys: "
        f"{sorted(d)[:20]} — pass the correct artifact or extend _extract_floor at fold-in"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="#1775 P6 noise-ceiling-relative gains")
    ap.add_argument("--noise-floor-json", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    floor_doc = json.loads(args.noise_floor_json.read_text())
    floor = _extract_floor(floor_doc)
    ceiling = 1.0 - floor if floor < 0.5 else floor  # floor may arrive as a ceiling
    nl_path = eval_dir("ladder") / "nonlinear_fits.json"
    if not nl_path.exists():
        raise SystemExit(f"{nl_path} absent — run P3 first")
    nl = json.loads(nl_path.read_text())
    out = {
        "meta": result_meta(smoke=args.smoke, noise_floor_source=str(args.noise_floor_json)),
        "noise_floor_raw": floor,
        "r2_ceiling_used": ceiling,
        "gains_noise_relative": {},
    }
    for key, g in (nl.get("gains_vs_ridge") or {}).items():
        denom = max(ceiling, 1e-9)
        out["gains_noise_relative"][key] = {
            "delta_r2_raw": g["delta_r2"],
            "delta_r2_over_ceiling": g["delta_r2"] / denom,
            "ci95_over_ceiling": [c / denom for c in g["ci95_cluster"]],
        }
    dest = eval_dir("") / "noise_relative.json"
    atomic_write_json(dest, out)
    print(f"[p6] wrote {dest} (ceiling={ceiling:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
