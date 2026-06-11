#!/usr/bin/env python3
"""#603 Phase 0 — materialize the frozen per-family input JSONs (CPU, VM).

Writes ONE JSON per behavior family to
``eval_results/issue_603/inputs/{family}_panel.json``:

- ``panel``: the 24 panel personas ``{name: system_prompt | null}``
  (``null`` == no system message, the ``no_system`` persona);
- ``probes``: the 20 probe questions the shifts are read on
  (fact: first 20 A-family courthouse probes, the ``_build_probes``
  slice rule from ``issue541_geometry_extract.py``; refusal/EM: the 20
  #551 generic questions from ``eval_results/issue_521/inputs/questions.json``);
- ``cells``: one entry per reused adapter (9 fact + 6 refusal + 6 EM)
  with the HF adapter repo + subfolder, the source persona, seed, and
  the inherited prior where it exists (#541 teacher priors; #518
  behavioral base rates as the secondary axis — the log-prob priors are
  Phase-1 work, ``scripts/issue603_source_prior.py``);
- ``probe_sha256`` + reproducibility ``meta``.

Asserts (plan #603 §4 step 2): panel size 24 per family; every source in
its panel; 20 probes per family; fact teacher priors match the values
read from ``phase0_prescreen/prior_screen.json``.

Run (VM)::

    uv run python scripts/issue603_inputs.py \
        --out-dir eval_results/issue_603/inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i603_inputs")

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_PROBES = 20

# Reused adapters (plan #603 §10, Hub-verified at plan time).
FACT_ADAPTER_REPO = "superkaiba1/explore-persona-space-overflow"
FACT_ARM_BY_TEACHER: dict[str, str] = {
    "marine_biologist": "arm_marine_biologist",
    "courthouse_architecture_historian": "arm_courthouse_architecture_historian",
    "wooden_furniture_carpenter": "arm_top_prior_wooden_furniture_carpenter",
}
FACT_SEEDS = (42, 137, 256)

I518_ADAPTER_REPO = "superkaiba1/explore-persona-space"
I518_SEED = 42

QUESTIONS_521 = PROJECT_ROOT / "eval_results" / "issue_521" / "inputs" / "questions.json"
PRIOR_SCREEN = (
    PROJECT_ROOT / "eval_results" / "issue_541" / "phase0_prescreen" / "prior_screen.json"
)
PREDICTOR_COMPARISON = {
    "refusal": PROJECT_ROOT
    / "eval_results"
    / "issue_518"
    / "refusal"
    / "_inputs"
    / "predictor_comparison.json",
    "em": PROJECT_ROOT
    / "eval_results"
    / "issue_518"
    / "em"
    / "_inputs"
    / "predictor_comparison.json",
}


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _meta() -> dict:
    return {
        "issue": 603,
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model_id": BASE_MODEL,
    }


def _probe_sha256(probes: list[str]) -> str:
    return hashlib.sha256(json.dumps(probes, ensure_ascii=False).encode()).hexdigest()


def _fact_family() -> dict:
    """Fact-teacher family: #541 24-persona panel + 20 A-family probes + 9 adapters."""
    import issue541_geometry_extract as ge

    panel_names = ge._load_panel(smoke=False)
    pool = ge._panel_pool(panel_names)
    probes = ge._build_probes(N_PROBES)

    prior_screen = json.loads(PRIOR_SCREEN.read_text())
    priors: dict[str, float] = prior_screen["priors"]

    cells = []
    for teacher, arm in FACT_ARM_BY_TEACHER.items():
        assert teacher in priors, f"teacher {teacher!r} missing from prior_screen priors"
        for seed in FACT_SEEDS:
            cells.append(
                {
                    "cell_id": f"fact_{teacher}_seed{seed}",
                    "family": "fact",
                    "source": teacher,
                    "seed": seed,
                    "adapter_repo": FACT_ADAPTER_REPO,
                    "adapter_subfolder": (
                        f"adapters/exp541-{arm}-on_policy_suppression_cn-seed{seed}"
                    ),
                    "prior_logprob": float(priors[teacher]),
                    "source_base_rate": None,
                }
            )
    return {"panel": pool, "probes": probes, "cells": cells}


def _i518_family(family: str) -> dict:
    """Refusal / EM family: #518 24-persona panel + 20 #551 generic questions."""
    if family == "refusal":
        from explore_persona_space.experiments.i518_refusal_conditions import (
            CID_TO_REFUSAL_PERSONA as cid_to_persona,
        )
        from explore_persona_space.experiments.i518_refusal_conditions import (
            CONDITIONS_BY_ID as conds,
        )
        from explore_persona_space.experiments.i518_refusal_conditions import (
            REFUSAL_SOURCES as sources,
        )
    elif family == "em":
        from explore_persona_space.experiments.i518_em_conditions import (
            CID_TO_EM_PERSONA as cid_to_persona,
        )
        from explore_persona_space.experiments.i518_em_conditions import (
            CONDITIONS_BY_ID as conds,
        )
        from explore_persona_space.experiments.i518_em_conditions import (
            EM_SOURCES as sources,
        )
    else:
        raise ValueError(f"unknown i518 family {family!r}")

    pool: dict[str, str | None] = {
        cid_to_persona[cid]: conds[cid].system_prompt for cid in sorted(conds)
    }
    probes: list[str] = json.loads(QUESTIONS_521.read_text())

    # Secondary IV: per-source behavioral base rate from the parent's
    # predictor-comparison frame (constant across that source's bystander
    # cells — asserted).
    pc = json.loads(PREDICTOR_COMPARISON[family].read_text())
    base_rates: dict[str, float] = {}
    for cell in pc["cells"]:
        s = cell["source"]
        r = float(cell["source_base_rate"])
        if s in base_rates and abs(base_rates[s] - r) > 1e-9:
            raise AssertionError(f"{family}: inconsistent source_base_rate for {s!r}")
        base_rates[s] = r

    cells = []
    for source in sources:
        assert source in base_rates, f"{family}: source {source!r} missing base rate"
        cells.append(
            {
                "cell_id": f"{family}_{source}_seed{I518_SEED}",
                "family": family,
                "source": source,
                "seed": I518_SEED,
                "adapter_repo": I518_ADAPTER_REPO,
                "adapter_subfolder": f"adapters/issue_518/{family}/{source}_seed{I518_SEED}",
                # Log-prob prior computed in Phase 1 (issue603_source_prior.py).
                "prior_logprob": None,
                "source_base_rate": base_rates[source],
            }
        )
    return {"panel": pool, "probes": probes, "cells": cells}


def main() -> int:
    """Build + assert + write the three frozen family input JSONs."""
    ap = argparse.ArgumentParser(description="Materialize #603 frozen per-family inputs")
    ap.add_argument("--out-dir", default="eval_results/issue_603/inputs")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "fact": _fact_family,
        "refusal": lambda: _i518_family("refusal"),
        "em": lambda: _i518_family("em"),
    }
    n_cells_total = 0
    for family, build in builders.items():
        spec = build()
        panel, probes, cells = spec["panel"], spec["probes"], spec["cells"]
        # Plan §4 step 2 asserts.
        assert len(panel) == 24, f"{family}: expected 24 panel personas, got {len(panel)}"
        assert len(probes) == N_PROBES, f"{family}: expected {N_PROBES} probes"
        for cell in cells:
            assert cell["source"] in panel, f"{family}: source {cell['source']!r} not in its panel"
        payload = {
            "family": family,
            "panel": panel,
            "probes": probes,
            "probe_sha256": _probe_sha256(probes),
            "cells": cells,
            "meta": _meta(),
        }
        path = out_dir / f"{family}_panel.json"
        with path.open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        n_cells_total += len(cells)
        logger.info(
            "[wrote] %s (%d personas, %d probes, %d cells, probe_sha=%s)",
            path,
            len(panel),
            len(probes),
            len(cells),
            payload["probe_sha256"][:12],
        )

    assert n_cells_total == 21, f"expected 21 cells total, got {n_cells_total}"
    logger.info("[done] 21 cells across 3 families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
