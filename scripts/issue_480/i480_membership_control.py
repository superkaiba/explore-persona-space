# ruff: noqa: RUF001, RUF002, RUF003  # research code uses Greek letters (ρ, Δ) legitimately
"""Task #480 — training-negative MEMBERSHIP common-cause control (committed).

Round 2 (`band-stopped-anchor-rerun`) computed this control INLINE at
interpretation: each source's training mix named two specific bystander
personas plus the no-system-prompt default as contrastive negatives, and
those cells could carry the marker↔sycophancy concordance by construction.
The inline analysis showed they don't. No committed script implemented it —
this file re-creates that analysis as a reusable artifact (round-3 plan §5
"Training-negative membership exclusion + third-partial re-read" row) and is
dry-run-validated against round 2's committed matrix before any production
read.

Two reads per source, both via the concordance script's OWN stats machinery
(imported, not re-implemented — "same partial machinery"):

1. EXCLUSION: drop the source's training-negative cells (typically 3 of 23)
   → naive Spearman + within-vector permutation p AND joint rank partial
   (cosine-L20 + bystander base rate) + permutation p on the kept cells.
2. THIRD-PARTIAL: keep all 23 cells, add the binary membership indicator as
   a third control column next to the two standard ones.

Membership derivation (validated this session by EXACT reproduction of all
six round-2 ρ point estimates):
  - the 2 named bystanders per source = the distinct non-source system
    prompts in the source's pinned 700-row train pool
    (``issue480_marker_payload_swap/train_pools/<source>_train_pool.jsonl``
    at revision ``3c8fecb9…``), mapped to panel names by exact prompt
    equality against ``EVAL_PERSONAS_24``;
  - the no-system-prompt negative rows map to the ``qwen_default`` panel
    persona: the Qwen-2.5-Instruct chat template inserts the qwen_default
    system prompt when no system message is present, so no-persona training
    contexts are token-identical to qwen_default-prompted contexts;
  - a source never appears in its own membership set (its panel excludes it).

Analysis-only: no training, no model loads, no GPU.

Production (round 3, plan §6 seeds):
    uv run python scripts/issue_480/i480_membership_control.py \
      --matrix-path eval_results/issue_480/inband-logprob-concordance/marker_delta_matrix.json \
      --x-field marker_delta \
      --out-path eval_results/issue_480/inband-logprob-concordance/\
        membership_control_marker_delta.json

Round-2 validation dry-run (the smoke; round 2's inline run used the
concordance script's seeds 4801/4802, so the dry-run passes them
explicitly):
    uv run python scripts/issue_480/i480_membership_control.py \
      --matrix-path eval_results/issue_480/band-stopped-anchor-rerun/marker_delta_matrix.json \
      --x-field emission_rate \
      --perm-seed 4801 --partial-perm-seed 4802 \
      --validate-round2 \
      --out-path /tmp/i480_membership_control_round2_dryrun.json
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Import the concordance script as a module so the stats machinery is shared
# byte-for-byte (same Spearman/permutation/partial implementations + the same
# matrix loader/schema asserts). scripts/ is not a package — path-insert it.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR))
import issue480_emission_rate_concordance as conc  # noqa: E402

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TRAIN_POOL_SUBDIR = "issue480_marker_payload_swap/train_pools"
TRAIN_POOL_REVISION = "3c8fecb937c81c13036a9697be1e4e716755321e"
# The no-system-prompt training rows render through the Qwen-2.5 chat
# template, which inserts the qwen_default system prompt when no system
# message is present — so their eval-panel twin is qwen_default.
NO_PERSONA_PANEL_NAME = "qwen_default"

# Plan §6: membership-control seeds for the round-3 production read.
DEFAULT_PERM_SEED = 4804
DEFAULT_PARTIAL_PERM_SEED = 4805
N_PERM = 100_000

DEFAULT_SOURCES = "software_engineer,assistant"

# Round-2 body-quoted values (X = emission_rate, band-stopped-anchor-rerun
# matrix) for --validate-round2. ρ point estimates are deterministic given
# the membership mapping; permutation p values are seed-dependent (round 2
# used seeds 4801/4802 for naive/partial; the third-partial p was quoted at
# 0.0036 for SE, which a 4802-seeded rerun reproduces at 0.0032 — within MC
# tolerance), so ρ is asserted tightly and p loosely.
ROUND2_EXPECTED = {
    "software_engineer": {
        "excl_naive_rho": 0.468,
        "excl_joint_partial_rho": 0.598,
        "third_partial_rho": 0.584,
        "excl_naive_p": 0.038,
        "excl_joint_partial_p": 0.0051,
        "third_partial_p": 0.0036,
    },
    "assistant": {
        "excl_naive_rho": 0.551,
        "excl_joint_partial_rho": 0.632,
        "third_partial_rho": 0.599,
        "excl_naive_p": 0.0127,
        "excl_joint_partial_p": 0.0031,
        "third_partial_p": 0.0024,
    },
}
RHO_TOLERANCE = 0.005
P_TOLERANCE = 0.02


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def derive_membership(sources: list[str]) -> dict[str, list[str]]:
    """Per-source training-negative panel cells from the pinned train pools.

    Scans each source's 700-row pool at the pinned revision: distinct
    non-source system prompts → panel names (exact prompt equality against
    ``EVAL_PERSONAS_24``); any no-persona rows add ``qwen_default``. The
    source itself is never a member of its own set.
    """
    from huggingface_hub import hf_hub_download

    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    prompt_to_name = {v: k for k, v in EVAL_PERSONAS_24.items()}
    membership: dict[str, list[str]] = {}
    for source in sources:
        if source not in EVAL_PERSONAS_24:
            raise KeyError(f"source {source!r} not in EVAL_PERSONAS_24")
        pool_path = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_TRAIN_POOL_SUBDIR}/{source}_train_pool.jsonl",
            repo_type="dataset",
            revision=TRAIN_POOL_REVISION,
        )
        src_prompt = EVAL_PERSONAS_24[source]
        bystander_prompts: set[str] = set()
        n_no_persona = 0
        with open(pool_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                first = row["prompt"][0]
                if first["role"] == "user":
                    n_no_persona += 1
                    continue
                if first["role"] != "system":
                    raise AssertionError(f"unexpected first role {first['role']!r} in {pool_path}")
                content = first["content"]
                if content != src_prompt:
                    bystander_prompts.add(content)
        unmapped = sorted(p[:60] for p in bystander_prompts if p not in prompt_to_name)
        if unmapped:
            raise RuntimeError(
                f"[{source}] {len(unmapped)} pool bystander prompts have no exact "
                f"EVAL_PERSONAS_24 match: {unmapped}"
            )
        cells = sorted(prompt_to_name[p] for p in bystander_prompts)
        if n_no_persona > 0:
            cells.append(NO_PERSONA_PANEL_NAME)
        cells = sorted(set(cells) - {source})
        if not cells:
            raise RuntimeError(f"[{source}] derived an EMPTY membership set — pool scan bug")
        membership[source] = cells
    return membership


def membership_reads(
    rows: list[dict],
    source: str,
    member_cells: list[str],
    x_field: str,
    perm_seed: int,
    partial_perm_seed: int,
    n_perm: int,
) -> dict:
    """Exclusion + third-partial membership reads for one source panel."""
    sub = [r for r in rows if r["source"] == source]
    if len(sub) != 23:
        raise RuntimeError(f"[{source}] expected 23 panel rows, got {len(sub)}")
    member_set = set(member_cells)
    unknown = member_set - {r["bystander"] for r in sub}
    if unknown:
        raise RuntimeError(f"[{source}] membership cells not in panel: {sorted(unknown)}")

    kept = [r for r in sub if r["bystander"] not in member_set]
    xk = np.array([r[x_field] for r in kept], dtype=float)
    yk = np.array([r[conc.Y_FIELD] for r in kept], dtype=float)
    ctrl_k = np.array([[r[c] for c in conc.CONTROL_FIELDS] for r in kept], dtype=float)
    excl_naive = conc.spearman_with_permutation(xk, yk, n_perm, perm_seed)
    excl_joint = conc.partial_spearman_with_permutation(xk, yk, ctrl_k, n_perm, partial_perm_seed)

    x23 = np.array([r[x_field] for r in sub], dtype=float)
    y23 = np.array([r[conc.Y_FIELD] for r in sub], dtype=float)
    member_col = np.array([1.0 if r["bystander"] in member_set else 0.0 for r in sub])
    ctrl_23 = np.column_stack(
        [
            np.array([r[conc.CONTROL_FIELDS[0]] for r in sub], dtype=float),
            np.array([r[conc.CONTROL_FIELDS[1]] for r in sub], dtype=float),
            member_col,
        ]
    )
    third = conc.partial_spearman_with_permutation(x23, y23, ctrl_23, n_perm, partial_perm_seed)
    naive_all = conc.spearman_with_permutation(x23, y23, n_perm, perm_seed)

    return {
        "membership_cells": sorted(member_set),
        "n_membership_cells": len(member_set),
        "n_kept_cells": len(kept),
        "all_cells_naive": naive_all,
        "exclusion_naive": excl_naive,
        "exclusion_joint_partial": excl_joint,
        "third_partial_with_membership": third,
        "partials_method": conc.PARTIAL_METHOD,
        "controls": [*conc.CONTROL_FIELDS, "training_negative_membership"],
    }


def _validate_round2(per_source: dict) -> list[str]:
    """Assert the rebuilt control reproduces round 2's body-quoted values."""
    failures: list[str] = []
    for source, exp in ROUND2_EXPECTED.items():
        got = per_source.get(source)
        if got is None:
            failures.append(f"{source}: missing from results")
            continue
        checks = [
            ("excl_naive_rho", got["exclusion_naive"]["rho"], RHO_TOLERANCE),
            (
                "excl_joint_partial_rho",
                got["exclusion_joint_partial"]["rho_partial"],
                RHO_TOLERANCE,
            ),
            (
                "third_partial_rho",
                got["third_partial_with_membership"]["rho_partial"],
                RHO_TOLERANCE,
            ),
            ("excl_naive_p", got["exclusion_naive"]["p_permutation"], P_TOLERANCE),
            ("excl_joint_partial_p", got["exclusion_joint_partial"]["p_permutation"], P_TOLERANCE),
            ("third_partial_p", got["third_partial_with_membership"]["p_permutation"], P_TOLERANCE),
        ]
        for key, observed, tol in checks:
            if abs(observed - exp[key]) > tol:
                failures.append(
                    f"{source}.{key}: observed {observed:.4f} vs round-2 {exp[key]:.4f} (tol {tol})"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matrix-path", type=Path, required=True)
    parser.add_argument(
        "--x-field",
        type=str,
        default="marker_delta",
        choices=sorted(conc.X_FIELD_DESCRIPTIONS),
        help="Marker-side DV column (round-3 primary: marker_delta; the round-2 "
        "validation dry-run passes emission_rate).",
    )
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument(
        "--sources",
        type=str,
        default=DEFAULT_SOURCES,
        help="Comma-separated source panels to read (default: the y-eligible set).",
    )
    parser.add_argument(
        "--membership-json",
        type=Path,
        default=None,
        help="Optional explicit {source: [panel_cells]} mapping; default derives it "
        "from the pinned train pools (the realized training mix).",
    )
    parser.add_argument("--perm-seed", type=int, default=DEFAULT_PERM_SEED)
    parser.add_argument("--partial-perm-seed", type=int, default=DEFAULT_PARTIAL_PERM_SEED)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument(
        "--validate-round2",
        action="store_true",
        help="Hard-assert the per-source reads reproduce round 2's body-quoted "
        "membership-control values (ρ within 0.005; permutation p within 0.02). "
        "Use with the round-2 matrix + --x-field emission_rate + seeds 4801/4802.",
    )
    args = parser.parse_args(argv)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows = conc.load_matrix(args.matrix_path, args.x_field)

    if args.membership_json is not None:
        with open(args.membership_json) as f:
            membership = {k: list(v) for k, v in json.load(f).items()}
    else:
        membership = derive_membership(sources)
    for source in sources:
        if source not in membership:
            raise KeyError(f"membership mapping missing source {source!r}")

    per_source: dict[str, dict] = {}
    for source in sources:
        per_source[source] = membership_reads(
            rows,
            source,
            membership[source],
            args.x_field,
            args.perm_seed,
            args.partial_perm_seed,
            args.n_perm,
        )
        e = per_source[source]
        naive, joint = e["exclusion_naive"], e["exclusion_joint_partial"]
        print(
            f"{source}: members={e['membership_cells']} | "
            f"all23 rho={e['all_cells_naive']['rho']:.3f} | "
            f"excl naive rho={naive['rho']:.3f} (p={naive['p_permutation']:.4f}) "
            f"joint={joint['rho_partial']:.3f} "
            f"(p={joint['p_permutation']:.4f}) | "
            f"third-partial rho={e['third_partial_with_membership']['rho_partial']:.3f} "
            f"(p={e['third_partial_with_membership']['p_permutation']:.4f})"
        )

    validation: dict | None = None
    if args.validate_round2:
        failures = _validate_round2(per_source)
        validation = {
            "expected": ROUND2_EXPECTED,
            "rho_tolerance": RHO_TOLERANCE,
            "p_tolerance": P_TOLERANCE,
            "failures": failures,
            "passed": not failures,
        }
        if failures:
            raise RuntimeError(
                "round-2 membership-control validation FAILED:\n  " + "\n  ".join(failures)
            )
        print("validate-round2: PASS (all ρ within 0.005, all permutation p within 0.02)")

    matrix_payload = json.loads(args.matrix_path.read_text())
    result = {
        "schema": "issue_480_membership_control_v1",
        "x_field": args.x_field,
        "y_field": conc.Y_FIELD,
        "input_matrix": {
            "path": str(args.matrix_path),
            "schema": matrix_payload["schema"],
            "git_commit_sha": matrix_payload["git_commit_sha"],
            "n_rows": matrix_payload["n_rows"],
        },
        "membership": {s: membership[s] for s in sources},
        "membership_derivation": (
            "explicit --membership-json"
            if args.membership_json is not None
            else f"pinned train pools {HF_TRAIN_POOL_SUBDIR}@{TRAIN_POOL_REVISION[:12]}; "
            f"no-persona rows -> {NO_PERSONA_PANEL_NAME} (Qwen chat-template default)"
        ),
        "seeds": {"permutation": args.perm_seed, "partial_permutation": args.partial_perm_seed},
        "n_perm": args.n_perm,
        "per_source": per_source,
        "validation_round2": validation,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env_versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"membership control -> {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
