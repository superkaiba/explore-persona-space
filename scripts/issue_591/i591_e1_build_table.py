#!/usr/bin/env python3
"""Task #591 e1 — assemble the 414-row cross-behavior cell table + 18-row panel table.

Joins the three frozen 138-cell leakage panels (sycophancy #411 via the #480
freeze, refusal + EM #518) into one cell table with the four registered
factors per cell (plan #591 v1 §4.1):

  - ``cos_to_source``  — PRIMARY: the syco join's Instruct-substrate
    ``cosine_l20_baseline`` joined onto all three behaviors by (source,
    bystander) pair (persona-pair geometry is behavior-independent; removes
    the #518 substrate swap from the isolation factor). The per-arm
    base-substrate cosine is kept as the robustness column.
  - ``self_delta``     — syco from the #411 ``analyze_summary.json``
    manipulation checks; refusal/EM from ``i591_judge_self_cells.py`` output.
  - ``bystander_base_rate`` — from each arm's join (behavior-specific).
  - ``neg_member``     — #411 parsed from the realized Hub training pools
    (system-prompt string matching); #518 reconstructed via the deterministic
    ``_draw_bystander_negs`` draw at pinned commit 4b150926 (flagged MEDIUM).

Inputs are snapshotted into ``eval_results/issue_591/_inputs/`` with a
provenance README (the #480 pattern) so the join cannot drift.

Usage (production, after i591_judge_self_cells.py):

    uv run python scripts/issue_591/i591_e1_build_table.py

Smoke (before self-rates exist — refusal/EM self_delta recorded as null):

    uv run python scripts/issue_591/i591_e1_build_table.py --allow-missing-self-rates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

OUT_ROOT_DEFAULT = REPO / "eval_results" / "issue_591"
TAU_PRIMARY = 0.10
TAU_SENSITIVITY = (0.05, 0.15, 0.20)

JOIN_PATHS = {
    "sycophancy": REPO / "eval_results/issue_480/_inputs/predictor_comparison.json",
    "refusal": REPO / "eval_results/issue_518/refusal/_inputs/predictor_comparison.json",
    "em": REPO / "eval_results/issue_518/em/_inputs/predictor_comparison.json",
}
EM_RUNS_DIR = REPO / "eval_results/issue_518/em/runs"
ANALYZE_SUMMARY_SNAPSHOT = OUT_ROOT_DEFAULT / "_inputs/issue411_analyze_summary.json"

SOURCES = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)

# The i509 24-persona registry NAMES at pinned commit 4b150926 (the #518
# bystander-negative draw universe — src/explore_persona_space/experiments/
# i509_syco_conditions.py::_SYCO_PERSONA_PROMPTS, which exists only on the
# issue-518 branch). Names only: the deterministic draw shuffles sorted names.
I509_PERSONA_NAMES_4B150926 = (
    "accountant",
    "ai",
    "ai_assistant",
    "assistant",
    "chef",
    "child",
    "comedian",
    "data_scientist",
    "french_person",
    "hero",
    "journalist",
    "kindergarten_teacher",
    "lawyer",
    "librarian",
    "medical_doctor",
    "philosopher",
    "police_officer",
    "programmer",
    "qwen_default",
    "software_engineer",
    "surgeon",
    "villain",
    "wizard",
    "zelthari_scholar",
)

# Expected realized #411 negative sets (verified from the Hub pools at plan
# time AND re-verified at implementation time). The pool parse below MUST
# reproduce these for the three isolated sources; a mismatch is a hard fail.
EXPECTED_411_NEGATIVES_SUBSET = {
    "villain": {"police_officer", "medical_doctor"},
    "comedian": {"medical_doctor", "assistant"},
    "kindergarten_teacher": {"software_engineer", "french_person"},
}

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
POOL_PATH_TMPL = (
    "issue411_sycophancy_cosine_gradient/training_pools/{source}_seed42/train_pool.jsonl"
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _stable_source_seed(source: str, seed: int) -> int:
    """sha256-anchored RNG seed — verbatim from 4b150926 run_experiment_518_*."""
    digest = hashlib.sha256(source.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") ^ int(seed)


def reconstruct_518_negatives(source: str, n_bystanders: int = 4, seed: int = 42) -> list[str]:
    """Replicate ``_draw_bystander_negs`` at pinned commit 4b150926.

    candidates = sorted(registry names) minus source; shuffle with the
    sha256-anchored seed; take the first ``n_bystanders``. Deterministic by
    construction (the original replaced Python ``hash()`` for exactly this
    reason). MEDIUM-confidence reconstruction (realized pools not uploaded);
    flagged in the output metadata per plan §12 assumption 8.
    """
    candidates = [p for p in sorted(I509_PERSONA_NAMES_4B150926) if p != source]
    rng = random.Random(_stable_source_seed(source, seed))
    rng.shuffle(candidates)
    return candidates[:n_bystanders]


def parse_411_negatives(inputs_dir: Path) -> dict[str, list[str]]:
    """Recover realized #411 negative membership from the Hub training pools.

    Pool rows carry ``prompt`` = chat-message list with NO persona-name field;
    membership is recovered by matching each row's system-prompt string
    against the canonical EVAL_PERSONAS_24 roster prompts. Asserts the #411
    composition (200 source + 200 x 2 negatives + 100 no-persona = 700 rows)
    and the expected realized sets for the three isolated sources.

    Writes the derived membership + per-pool sha256 into
    ``inputs_dir / neg_membership_411.json`` and returns {source: [negs]}.
    """
    import os

    from huggingface_hub import hf_hub_download

    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    prompt_to_name = {v: k for k, v in EVAL_PERSONAS_24.items()}
    assert len(prompt_to_name) == 24, "EVAL_PERSONAS_24 prompts are not unique"

    out: dict[str, list[str]] = {}
    provenance: dict[str, dict] = {}
    for source in SOURCES:
        local = hf_hub_download(
            HF_DATA_REPO,
            POOL_PATH_TMPL.format(source=source),
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        raw = Path(local).read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        counts: dict[str, int] = {}
        for r in rows:
            sys_content = next(
                (m["content"] for m in r["prompt"] if m.get("role") == "system"), None
            )
            if sys_content is None:
                key = "<none>"
            else:
                name = prompt_to_name.get(sys_content)
                if name is None:
                    raise ValueError(
                        f"pool row system prompt not in EVAL_PERSONAS_24 roster "
                        f"(source={source}): {sys_content[:80]!r}"
                    )
                key = name
            counts[key] = counts.get(key, 0) + 1
        if len(rows) != 700 or counts.get("<none>") != 100 or counts.get(source) != 200:
            raise AssertionError(f"unexpected #411 pool composition for {source}: {counts}")
        negs = sorted(k for k in counts if k not in ("<none>", source))
        if len(negs) != 2 or any(counts[n] != 200 for n in negs):
            raise AssertionError(f"expected 2 negatives x 200 rows for {source}, got {counts}")
        expected = EXPECTED_411_NEGATIVES_SUBSET.get(source)
        if expected is not None and set(negs) != expected:
            raise AssertionError(
                f"realized #411 negatives for {source} = {negs}, expected {sorted(expected)}"
            )
        out[source] = negs
        provenance[source] = {
            "pool_sha256": hashlib.sha256(raw).hexdigest(),
            "n_rows": len(rows),
            "composition": counts,
        }
    payload = {
        "negatives_by_source": out,
        "provenance": provenance,
        "recovery": "system-prompt string match vs EVAL_PERSONAS_24 (plan §12 assumption 9)",
        "hf_repo": HF_DATA_REPO,
        "git_commit_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "neg_membership_411.json").write_text(json.dumps(payload, indent=2))
    return out


def _load_join(behavior: str) -> list[dict]:
    d = json.loads(JOIN_PATHS[behavior].read_text())
    cells = d["cells"]
    assert d["n_cells"] == 138 and len(cells) == 138, (behavior, d.get("n_cells"), len(cells))
    return cells


def _load_em_survivors() -> dict[tuple[str, str], dict[str, int]]:
    """Per-(source, bystander) EM coherence-survivor counts (#518 run_result)."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    files = sorted(EM_RUNS_DIR.glob("*_seed42/run_result.json"))
    assert len(files) == 6, f"expected 6 EM run_result.json files, found {len(files)}"
    for f in files:
        d = json.loads(f.read_text())
        for cell in d["per_cell"]:
            key = (cell["source"], cell["bystander"])
            out[key] = {
                "n_rollouts_after_coherence_filter": cell["n_rollouts_after_coherence_filter"],
                "n_rollouts_total": cell["n_rollouts_total"],
            }
    assert len(out) == 138, f"expected 138 EM survivor cells, got {len(out)}"
    return out


def _assert_join_schemas(joins: dict[str, list[dict]]) -> None:
    """Schema asserts for the three joins (plan §12 assumption 1)."""
    key_sets = {b: {(c["source"], c["bystander"]) for c in cells} for b, cells in joins.items()}
    assert key_sets["sycophancy"] == key_sets["refusal"] == key_sets["em"], (
        "(source, bystander) keys differ across the three joins"
    )
    syco_fields = set(joins["sycophancy"][0].keys())
    assert "trained_rate_411" in syco_fields and "completion_logprob" not in syco_fields, (
        "syco join schema drifted from the expected field-name differences"
    )
    for b in ("refusal", "em"):
        f = set(joins[b][0].keys())
        assert "trained_rate" in f and "completion_logprob" in f, f"{b} join schema drifted"


def _load_self_deltas(
    self_rates_path: Path, allow_missing: bool
) -> tuple[dict[tuple[str, str], float | None], dict]:
    """syco self deltas from #411 analyze_summary; refusal/EM from self_rates.json."""
    analyze_summary = json.loads(ANALYZE_SUMMARY_SNAPSHOT.read_text())
    self_delta: dict[tuple[str, str], float | None] = {}
    for s in SOURCES:
        self_delta[("sycophancy", s)] = analyze_summary["per_source"][s]["self_delta"]
    meta: dict = {"path": str(self_rates_path), "present": self_rates_path.exists()}
    if self_rates_path.exists():
        sr = json.loads(self_rates_path.read_text())
        for arm in ("refusal", "em"):
            for s in SOURCES:
                self_delta[(arm, s)] = sr["arms"][arm][s]["self_delta"]
        meta["n_completions_judged"] = {
            arm: {s: sr["arms"][arm][s].get("n_total") for s in SOURCES}
            for arm in ("refusal", "em")
        }
    elif allow_missing:
        for arm in ("refusal", "em"):
            for s in SOURCES:
                self_delta[(arm, s)] = None
    else:
        raise FileNotFoundError(
            f"{self_rates_path} missing — run i591_judge_self_cells.py first, or pass "
            f"--allow-missing-self-rates (refusal/EM self_delta recorded as null)."
        )
    return self_delta, meta


def _membership(inputs_dir: Path) -> tuple[dict, dict, dict]:
    """Realized #411 membership + deterministic #518 reconstruction + lookup."""
    membership_411_path = inputs_dir / "neg_membership_411.json"
    if membership_411_path.exists():
        negatives_411 = json.loads(membership_411_path.read_text())["negatives_by_source"]
    else:
        negatives_411 = parse_411_negatives(inputs_dir)
    negatives_518 = {s: reconstruct_518_negatives(s) for s in SOURCES}
    (inputs_dir / "neg_membership_518.json").write_text(
        json.dumps(
            {
                "negatives_by_source": negatives_518,
                "recovery": (
                    "deterministic _draw_bystander_negs replication at pinned commit "
                    "4b150926 (sha256-seeded shuffle of the i509 24-name registry); "
                    "realized pools were not uploaded — MEDIUM confidence "
                    "(plan §12 assumption 8); H4 primary read stays on #411"
                ),
                "registry_names": list(I509_PERSONA_NAMES_4B150926),
                "git_commit_sha": _git_sha(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    neg_member: dict[tuple[str, str, str], int] = {}
    for s in SOURCES:
        for b in negatives_411[s]:
            neg_member[("sycophancy", s, b)] = 1
        for arm in ("refusal", "em"):
            for b in negatives_518[s]:
                neg_member[(arm, s, b)] = 1
    return negatives_411, negatives_518, neg_member


def build(out_root: Path, self_rates_path: Path, allow_missing_self_rates: bool) -> dict:
    """Assemble cell + panel tables; snapshot inputs; return the payload dict."""
    import numpy as np

    inputs_dir = out_root / "_inputs"
    e1_dir = out_root / "e1"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    e1_dir.mkdir(parents=True, exist_ok=True)

    # ---- snapshot the three joins (the #480 _inputs pattern) ----
    for behavior, src_path in JOIN_PATHS.items():
        shutil.copy2(src_path, inputs_dir / f"join_{behavior}.json")

    joins = {b: _load_join(b) for b in JOIN_PATHS}
    _assert_join_schemas(joins)

    # ---- Instruct-substrate cosine lookup (PRIMARY isolation factor) ----
    cos_instruct = {
        (c["source"], c["bystander"]): c["cosine_l20_baseline"] for c in joins["sycophancy"]
    }

    self_delta, self_rates_meta = _load_self_deltas(self_rates_path, allow_missing_self_rates)
    negatives_411, negatives_518, neg_member = _membership(inputs_dir)

    em_survivors = _load_em_survivors()

    # ---- cell table ----
    cells_out: list[dict] = []
    for behavior, cells in joins.items():
        for c in cells:
            s, b = c["source"], c["bystander"]
            trained_rate = c.get("trained_rate", c.get("trained_rate_411"))
            row = {
                "behavior": behavior,
                "source": s,
                "bystander": b,
                "delta": c["delta"],
                "leak": int(c["delta"] >= TAU_PRIMARY),
                "cos_to_source": cos_instruct[(s, b)],
                "cos_arm_substrate": c["cosine_l20_baseline"],
                "self_delta": self_delta[(behavior, s)],
                "bystander_base_rate": c["bystander_base_rate"],
                "source_base_rate": c["source_base_rate"],
                "neg_member": neg_member.get((behavior, s, b), 0),
                "trained_rate": trained_rate,
                "resp_len_diff_abs": c.get("resp_len_diff_abs"),
                "completion_logprob": c.get("completion_logprob"),
            }
            if behavior == "em":
                row.update(em_survivors[(s, b)])
            cells_out.append(row)
    assert len(cells_out) == 414, len(cells_out)

    # ---- 18-row panel table (H2's inferential substrate) ----
    panels_out: list[dict] = []
    for behavior, cells in joins.items():
        for s in SOURCES:
            deltas = [c["delta"] for c in cells if c["source"] == s]
            coses = [cos_instruct[(s, c["bystander"])] for c in cells if c["source"] == s]
            assert len(deltas) == 23, (behavior, s, len(deltas))
            n_leak = sum(1 for d in deltas if d >= TAU_PRIMARY)
            panels_out.append(
                {
                    "behavior": behavior,
                    "source": s,
                    "self_delta": self_delta[(behavior, s)],
                    "n_leak_cells": n_leak,
                    "panel_sd": float(np.std(deltas, ddof=1)),
                    "max_bystander_cos": float(max(coses)),
                    "flat": int(n_leak == 0),
                }
            )
    assert len(panels_out) == 18, len(panels_out)

    payload = {
        "cells": cells_out,
        "panels": panels_out,
        "tau_primary": TAU_PRIMARY,
        "tau_sensitivity": list(TAU_SENSITIVITY),
        "negatives_411": negatives_411,
        "negatives_518": negatives_518,
        "self_rates": self_rates_meta,
        "confounds": {
            "substrate_swap": (
                "PRIMARY cosine = single Instruct-substrate bank joined to all three "
                "behaviors (neutralizes #518's predictor-extraction substrate swap); "
                "per-arm base-substrate cosine kept as cos_arm_substrate robustness "
                "column; KL/JS fields remain substrate-confounded and are NOT factors"
            ),
            "em_survivorship": (
                "EM DV is the coherence-filter survivor rate; per-cell survivor counts "
                "carried (n_rollouts_after_coherence_filter / n_rollouts_total); "
                "sensitivity filter at <24 survivors in the factor analysis"
            ),
            "refusal_floor": "76% of refusal cells |delta| < 0.02 (#518) — power-limited",
            "single_seed": "all panels seed 42 only — confidence cap inherited",
            "decoder_caveat_vacated": (
                "all three panels share temp 1.0 / max_new_tokens 512 (the #518 "
                "drift caveat traced to #411's body-table transcription error); "
                "decoder drift is NOT a cross-behavior confound"
            ),
            "neg_518_reconstructed": (
                "#518 membership reconstructed from the deterministic draw at 4b150926 "
                "(MEDIUM confidence); #518's realized panels include sources as "
                "negatives for other sources (e.g. kindergarten_teacher draws villain "
                "+ qwen_default) — the #527-class overlap, named not fixable; H4 "
                "primary read stays on #411 realized pools"
            ),
        },
        "metadata": {
            "git_commit_sha": _git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "numpy_version": np.__version__,
            "join_paths": {k: str(v) for k, v in JOIN_PATHS.items()},
        },
    }
    out_path = e1_dir / "cell_table.json"
    out_path.write_text(json.dumps(payload, indent=2))

    readme = inputs_dir / "README.md"
    readme.write_text(
        "# #591 e1 input snapshots (provenance)\n\n"
        "| file | producer | source path |\n|---|---|---|\n"
        "| join_sycophancy.json | #411→#470→#480 freeze | "
        "eval_results/issue_480/_inputs/predictor_comparison.json (git, main) |\n"
        "| join_refusal.json | #518 | "
        "eval_results/issue_518/refusal/_inputs/predictor_comparison.json (git, main) |\n"
        "| join_em.json | #518 | "
        "eval_results/issue_518/em/_inputs/predictor_comparison.json (git, main) |\n"
        "| issue411_analyze_summary.json | #411 | "
        "origin/issue-411:eval_results/issue_411/analyze_summary.json |\n"
        "| issue411_base_panel_rates.json | #411 | "
        "origin/issue-411:eval_results/issue_411/base_panel_rates.json |\n"
        "| neg_membership_411.json | derived | Hub training pools "
        "(system-prompt match), sha256 per pool inside |\n"
        "| neg_membership_518.json | derived | deterministic draw @ 4b150926 "
        "(MEDIUM confidence) |\n"
        "| sex2.csv | vendored | firthlogist repo (Firth validation dataset) |\n"
        "| firth_sex2_validation.json | i591_firth.py --validate | this repo |\n\n"
        f"Snapshot commit: {_git_sha()} at {datetime.now(UTC).isoformat()}\n"
    )
    print(f"[e1-build] cell table -> {out_path} ({len(cells_out)} cells, {len(panels_out)} panels)")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the #591 e1 cross-behavior cell + panel tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    parser.add_argument(
        "--self-rates",
        type=Path,
        default=None,
        help="Path to i591_judge_self_cells.py output (default <out-root>/e1/self_rates.json).",
    )
    parser.add_argument(
        "--allow-missing-self-rates",
        action="store_true",
        help="Proceed with null refusal/EM self_delta when self_rates.json is absent (smoke).",
    )
    args = parser.parse_args(argv)
    self_rates = args.self_rates or (args.out_root / "e1" / "self_rates.json")
    build(args.out_root, self_rates, args.allow_missing_self_rates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
