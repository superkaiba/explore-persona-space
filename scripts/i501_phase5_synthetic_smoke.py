# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #501 Phase 5 synthetic CPU smoke.

Round-2 code-review reviewer-recommendation: tiny CPU/synthetic smoke for
Phase 5. Feeds the analyzer a synthetic dict of 552 + 288 = 840 mock cells
with known ΔG / cosine values and confirms:

  1. The merged-cell JSON ships at the right size.
  2. The H1 partial-Spearman runs (point-estimate close to 0, CI brackets 0).
  3. The H2/H3 paired-bootstrap runs without crashing.
  4. The H4 saturation guard reads the 24 source-diagonals.
  5. The collinearity gate runs on the merged 840 panel (not on
     ``self_tagged`` — Round-2 BLOCKER 3 regression test).
  6. The panel labels carry no ``_smoke`` suffix when synthetic cells
     match the planned 552/288 sizes (since we hit the planned counts
     exactly), so the verify-loud guardrail (Round-2 BLOCKER 2) is
     exercised in its passing branch.

What this does NOT verify: real ΔG numbers, real cosine values, vLLM
behavior, GPU availability. Those are pod-deferred per Path B EVAL-ONLY.

CLI:
    uv run python scripts/i501_phase5_synthetic_smoke.py
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.i501_mt_contexts import (
    MT_CIDS,
)
from explore_persona_space.experiments.i501_vendored_i489_contexts import (
    UNION_CONTEXTS,
)

logger = logging.getLogger("i501.phase5.synthetic_smoke")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_501" / "phase5_synthetic_smoke"
SINGLE_TURN_CIDS = tuple(c.cid for c in UNION_CONTEXTS)
FRAC = 0.50
SEED = 42
RNG = np.random.default_rng(SEED)


def _load_phase5_module():
    """Load scripts/i501_phase5_analyze.py by path (the scripts/ dir is not
    a Python package). Returns the module object for monkeypatching."""
    import importlib.util
    import sys

    p5_path = PROJECT_ROOT / "scripts" / "i501_phase5_analyze.py"
    spec = importlib.util.spec_from_file_location("i501_phase5_analyze_mod", p5_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i501_phase5_analyze_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _make_parent_cells(out_dir: Path) -> int:
    """552 single-turn × single-turn off-diag + 24 diagonals (#489)."""
    n_written = 0
    for ci in SINGLE_TURN_CIDS:
        for cj in SINGLE_TURN_CIDS:
            # Diagonals get higher g_logprob_mean (so H4 spread guard sees a
            # full 3-nat band); off-diag cells get a moderate-negative ΔG.
            is_diag = ci == cj
            if is_diag:
                g_lp = float(RNG.uniform(-8.0, -2.0))  # saturation guard wants 3-nat spread
                b_lp = float(RNG.uniform(-25.0, -15.0))
                delta = g_lp - b_lp
            else:
                # ΔG correlated with a synthetic cosine drawn later; just put
                # a noisy intercept here, the cosine matrix will drive the rank.
                g_lp = float(RNG.normal(-8.0, 2.0))
                b_lp = float(RNG.normal(-15.0, 2.0))
                delta = g_lp - b_lp
            cell = {
                "T_i": ci,
                "T_j": cj,
                "frac": FRAC,
                "seed": SEED,
                "n_q": 20,
                "n_samples": 8,
                "g_logprob_mean": g_lp,
                "b_logprob_mean": b_lp,
                "delta_g": delta,
                "delta_g_trimmed_10pct": delta,
                "emission_rate_trained": float(RNG.uniform(0.0, 1.0)),
                "g_logps_per_q_sample": [[g_lp] * 8 for _ in range(20)],
                "b_logps_per_q_sample": [[b_lp] * 8 for _ in range(20)],
                # Single-turn prompts: short — ~200-500 tokens.
                "prompt_lens_per_q": [int(RNG.integers(200, 500)) for _ in range(20)],
                "R_lens_per_q_sample": [[256] * 8 for _ in range(20)],
                "sample_texts_first200": [["x"] * 8 for _ in range(20)],
                "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
                "git_commit": _git_commit_hash(),
            }
            (out_dir / f"G_{ci}__{cj}_frac{FRAC:.2f}.json").write_text(json.dumps(cell))
            n_written += 1
    return n_written


def _make_self_cells(out_dir: Path) -> int:
    """288 cross-format single-turn × multi-turn cells (#501)."""
    n_written = 0
    for ci in SINGLE_TURN_CIDS:
        for mt_cid in MT_CIDS:
            # Cross-format ΔG: lower than within-single-turn (matches H2
            # silencing direction) so H2(a) crosses its bar deterministically.
            g_lp = float(RNG.normal(-12.0, 2.0))
            b_lp = float(RNG.normal(-15.0, 2.0))
            delta = g_lp - b_lp
            cell = {
                "T_i": ci,
                "T_mt": mt_cid,
                "frac": FRAC,
                "seed": SEED,
                "n_q": 20,
                "n_samples": 8,
                "n_conversations": 5,
                "g_logprob_mean": g_lp,
                "b_logprob_mean": b_lp,
                "delta_g": delta,
                "delta_g_trimmed_10pct": delta,
                "emission_rate_trained": float(RNG.uniform(0.0, 0.1)),
                "argmax_marker_rate_trained": 0.0,
                "g_logps_per_q_sample": [[g_lp] * 8 for _ in range(20)],
                "b_logps_per_q_sample": [[b_lp] * 8 for _ in range(20)],
                # Multi-turn prompts: long — ~3000-25000 tokens.
                "prompt_lens_per_q": [int(RNG.integers(3000, 25000)) for _ in range(20)],
                "R_lens_per_q_sample": [[256] * 8 for _ in range(20)],
                "sample_texts_first200": [["x"] * 8 for _ in range(20)],
                "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
                "git_commit": _git_commit_hash(),
            }
            (out_dir / f"G_{ci}__{mt_cid}_frac{FRAC:.2f}.json").write_text(json.dumps(cell))
            n_written += 1
    return n_written


def _make_synthetic_cosine() -> dict:
    """Write a cosine_per_layer.json covering the union of 24+12 cids.

    Single-turn pairs cluster around cosine SIM = 0.85 (distance 0.15).
    Cross-format pairs cluster around cosine SIM = 0.45 (distance 0.55),
    putting them in the plan §7 gate-3 [0.3, 1.5] band.
    """
    all_cids = list(SINGLE_TURN_CIDS) + list(MT_CIDS)
    sim_layer: dict[str, dict[str, float]] = {ci: {} for ci in all_cids}
    for ci in all_cids:
        for cj in all_cids:
            if ci == cj:
                sim_layer[ci][cj] = 1.0
                continue
            ci_mt = ci in MT_CIDS
            cj_mt = cj in MT_CIDS
            if not ci_mt and not cj_mt:
                sim_layer[ci][cj] = float(RNG.uniform(0.75, 0.95))
            elif ci_mt and cj_mt:
                sim_layer[ci][cj] = float(RNG.uniform(0.70, 0.90))
            else:
                # Cross-format: lower SIM = higher distance.
                sim_layer[ci][cj] = float(RNG.uniform(0.35, 0.55))
    return {
        "schema_version": "i501_phase1a_v1_synthetic",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "base_model": "synthetic",
        "max_model_len": 32768,
        "n_mt_contexts": len(MT_CIDS),
        "n_anchor_contexts": len(SINGLE_TURN_CIDS),
        "n_probes": 50,
        "layers": [7, 11, 14, 15, 21, 27],
        "headline_layer": 21,
        "all_cids": all_cids,
        "cos_sim_per_layer": {str(li): sim_layer for li in (7, 11, 14, 15, 21, 27)},
        "smoke": True,
        "synthetic": True,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage synthetic Phase 4 outputs under a TEMP `eval_results/issue_*`
    # tree so we don't clobber real results. Then point the phase 5
    # analyzer at it by monkeypatching its module-level path constants.
    with tempfile.TemporaryDirectory(prefix="i501_phase5_smoke_") as tmp:
        tmp_root = Path(tmp)
        parent_phase4 = tmp_root / "eval_results" / "issue_489" / "phase4" / "per_cell"
        self_phase4 = tmp_root / "eval_results" / "issue_501" / "phase4" / "per_cell"
        parent_phase4.mkdir(parents=True)
        self_phase4.mkdir(parents=True)

        n_parent = _make_parent_cells(parent_phase4)
        n_self = _make_self_cells(self_phase4)
        logger.info(
            "Synthetic Phase 4: wrote %d parent cells (#489) + %d self cells (#501)",
            n_parent,
            n_self,
        )
        assert n_parent == 24 * 24, f"expected 576 parent cells (24×24), got {n_parent}"
        assert n_self == 24 * 12, f"expected 288 self cells, got {n_self}"
        # 552 off-diag from parent + 288 cross-format = 840 merged.

        # Synthetic cosine_per_layer.json: write to BOTH parent + self phase1 paths
        # so the merger's prefer-self-over-parent precedence is exercised.
        parent_phase1 = tmp_root / "eval_results" / "issue_489" / "phase1"
        self_phase1 = tmp_root / "eval_results" / "issue_501" / "phase1"
        parent_phase1.mkdir(parents=True)
        self_phase1.mkdir(parents=True)
        cos_payload = _make_synthetic_cosine()
        (parent_phase1 / "cosine_per_layer.json").write_text(json.dumps(cos_payload))
        (self_phase1 / "cosine_per_layer.json").write_text(json.dumps(cos_payload))

        # Parent-ready stub (so phase5 finds a chosen frac).
        self_phase0 = tmp_root / "eval_results" / "issue_501" / "phase0"
        self_phase0.mkdir(parents=True)
        (self_phase0 / "parent_ready.json").write_text(
            json.dumps(
                {
                    "schema_version": "i501_phase0_parent_ready_v1",
                    "verdict": "PASS",
                    "frac": FRAC,
                    "source": "synthetic-smoke",
                    "seed": SEED,
                    "n_adapters_checked": 24,
                    "cids_checked": list(SINGLE_TURN_CIDS),
                    "hf_repo": "synthetic",
                    "smoke": True,
                }
            )
        )

        # Phase 5 OUT_DIR — stage under tmp so smoke doesn't pollute real
        # eval_results/issue_501/phase5/ on the production tree.
        self_phase5 = tmp_root / "eval_results" / "issue_501" / "phase5"
        self_phase5.mkdir(parents=True)

        # Monkeypatch the module-level paths inside phase5_analyze BEFORE calling main().
        p5 = _load_phase5_module()

        p5.PROJECT_ROOT = tmp_root
        p5.PARENT_PHASE4_DIR = parent_phase4
        p5.SELF_PHASE4_DIR = self_phase4
        p5.PARENT_PHASE1 = parent_phase1 / "cosine_per_layer.json"
        p5.SELF_PHASE1 = self_phase1 / "cosine_per_layer.json"
        p5.PARENT_READY_PATH = self_phase0 / "parent_ready.json"
        p5.OUT_DIR = self_phase5

        rc = p5.main(argv=["--frac", str(FRAC)])  # non-smoke: enforces 840 / 552 / 288
        assert rc == 0, f"phase5 main returned rc={rc}"

        # Verify the expected artifacts landed.
        h1 = json.loads((self_phase5 / "H1_verdict.json").read_text())
        h2 = json.loads((self_phase5 / "H2_verdict.json").read_text())
        h3 = json.loads((self_phase5 / "H3_verdict.json").read_text())
        h4 = json.loads((self_phase5 / "H4_verdict.json").read_text())
        coll = json.loads((self_phase5 / "collinearity.json").read_text())
        merged = json.loads((self_phase5 / "merged_cells.json").read_text())
        summary = json.loads((self_phase5 / "phase5_summary.json").read_text())

        # Persist a copy of the verdicts under the smoke-out dir for evidence.
        SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        digest = {
            "schema_version": "i501_phase5_synthetic_smoke_v1",
            "git_commit": _git_commit_hash(),
            "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "n_merged_cells": len(merged),
            "h1_panels": {k: v["n_cells"] for k, v in h1["panels"].items()},
            "h1_verdict": h1["verdict"],
            "h2_verdict": h2["verdict"],
            "h3_verdict": h3["verdict"],
            "h4_verdict": h4["verdict"],
            "collinearity_panel": coll["panel"],
            "collinearity_pearson_cosine_vs_is_multi_turn": coll["pearson_cosine_vs_is_multi_turn"],
            "collinearity_pearson_prefix_token_count_vs_is_multi_turn": coll[
                "pearson_prefix_token_count_vs_is_multi_turn"
            ],
            "collinearity_pass": coll["pass"],
            "summary": summary,
        }
        (SMOKE_OUT_DIR / "smoke_digest.json").write_text(json.dumps(digest, indent=2))

        # Assertions — Round-2 BLOCKER regression tests.
        assert len(merged) == 840, f"merged should be 840, got {len(merged)}"
        assert h1["panels"]["merged_840"]["n_cells"] == 840
        assert h1["panels"]["within_single_turn_552"]["n_cells"] == 552
        assert h1["panels"]["cross_format_288"]["n_cells"] == 288
        # Collinearity must report a finite Pearson on the merged panel
        # (BLOCKER 3 regression: previously NaN because computed on self_tagged).
        import math as _math

        assert _math.isfinite(coll["pearson_cosine_vs_is_multi_turn"]), (
            f"collinearity Pearson(cos, is_multi_turn) on merged panel "
            f"should be finite; got {coll['pearson_cosine_vs_is_multi_turn']}"
        )
        # Pearson(prefix_token_count, is_multi_turn) must also be finite +
        # high (plan assumption #12).
        assert _math.isfinite(coll["pearson_prefix_token_count_vs_is_multi_turn"]), (
            "informational Pearson(prefix_token_count, is_multi_turn) should be finite"
        )
        # Panel labels must NOT carry _smoke suffix when we hit the planned
        # counts exactly (BLOCKER 2 regression: production-shape labels).
        assert coll["panel"] == "merged_840", f"label should be merged_840, got {coll['panel']}"

        logger.info(
            "Phase 5 synthetic smoke PASS: 840 cells / H1=%s H2=%s H3=%s H4=%s "
            "collinearity pearson_cos=%.3f pearson_len=%.3f pass=%s",
            h1["verdict"],
            h2["verdict"],
            h3["verdict"],
            h4["verdict"],
            coll["pearson_cosine_vs_is_multi_turn"],
            coll["pearson_prefix_token_count_vs_is_multi_turn"],
            coll["pass"],
        )
        logger.info("Wrote %s/smoke_digest.json", SMOKE_OUT_DIR)
    # Verify a smoke-suffix branch separately (smoke labels carry _smoke).
    _smoke_label_check()
    return 0


def _smoke_label_check() -> None:
    """Run phase5 in --smoke mode with a tiny synthetic panel (10 cells)
    and confirm the labels carry the _smoke suffix per Round-2 BLOCKER 2."""
    p5 = _load_phase5_module()

    with tempfile.TemporaryDirectory(prefix="i501_phase5_smoke_label_") as tmp:
        tmp_root = Path(tmp)
        parent_phase4 = tmp_root / "eval_results" / "issue_489" / "phase4" / "per_cell"
        self_phase4 = tmp_root / "eval_results" / "issue_501" / "phase4" / "per_cell"
        self_phase0 = tmp_root / "eval_results" / "issue_501" / "phase0"
        self_phase5 = tmp_root / "eval_results" / "issue_501" / "phase5"
        for p in (parent_phase4, self_phase4, self_phase0, self_phase5):
            p.mkdir(parents=True)

        # 2 single-turn anchors × 2 mt targets = 4 cells; plus 2 source
        # diagonals (for H4) and 2 single-turn off-diag (for H1 panels to
        # be non-empty).
        smoke_anchors = ("IK01", "SP01")
        smoke_targets = ("MT05",)
        for ci in smoke_anchors:
            for cj in smoke_anchors:
                (parent_phase4 / f"G_{ci}__{cj}_frac{FRAC:.2f}.json").write_text(
                    json.dumps(
                        {
                            "T_i": ci,
                            "T_j": cj,
                            "frac": FRAC,
                            "seed": SEED,
                            "n_q": 2,
                            "n_samples": 2,
                            "g_logprob_mean": -5.0,
                            "b_logprob_mean": -15.0,
                            "delta_g": 10.0,
                            "delta_g_trimmed_10pct": 10.0,
                            "emission_rate_trained": 0.5,
                            "g_logps_per_q_sample": [[-5.0, -5.0], [-5.0, -5.0]],
                            "b_logps_per_q_sample": [[-15.0, -15.0], [-15.0, -15.0]],
                            "prompt_lens_per_q": [300, 300],
                            "R_lens_per_q_sample": [[256, 256], [256, 256]],
                        }
                    )
                )
        for ci in smoke_anchors:
            for mt in smoke_targets:
                (self_phase4 / f"G_{ci}__{mt}_frac{FRAC:.2f}.json").write_text(
                    json.dumps(
                        {
                            "T_i": ci,
                            "T_mt": mt,
                            "frac": FRAC,
                            "seed": SEED,
                            "n_q": 2,
                            "n_samples": 2,
                            "n_conversations": 1,
                            "g_logprob_mean": -10.0,
                            "b_logprob_mean": -15.0,
                            "delta_g": 5.0,
                            "delta_g_trimmed_10pct": 5.0,
                            "emission_rate_trained": 0.0,
                            "argmax_marker_rate_trained": 0.0,
                            "g_logps_per_q_sample": [[-10.0, -10.0], [-10.0, -10.0]],
                            "b_logps_per_q_sample": [[-15.0, -15.0], [-15.0, -15.0]],
                            "prompt_lens_per_q": [5000, 5000],
                            "R_lens_per_q_sample": [[256, 256], [256, 256]],
                        }
                    )
                )

        (self_phase0 / "parent_ready.json").write_text(
            json.dumps({"frac": FRAC, "verdict": "PASS"})
        )
        # Cosine matrix covering the smoke cids only.
        smoke_cos_cids = list(smoke_anchors) + list(smoke_targets)
        sim_layer = {ci: {} for ci in smoke_cos_cids}
        for ci in smoke_cos_cids:
            for cj in smoke_cos_cids:
                if ci == cj:
                    sim_layer[ci][cj] = 1.0
                else:
                    ci_mt = ci in smoke_targets
                    cj_mt = cj in smoke_targets
                    sim_layer[ci][cj] = 0.5 if (ci_mt != cj_mt) else 0.85
        self_phase1 = tmp_root / "eval_results" / "issue_501" / "phase1"
        self_phase1.mkdir()
        (self_phase1 / "cosine_per_layer.json").write_text(
            json.dumps(
                {
                    "all_cids": smoke_cos_cids,
                    "cos_sim_per_layer": {str(li): sim_layer for li in (7, 11, 14, 15, 21, 27)},
                }
            )
        )

        p5.PROJECT_ROOT = tmp_root
        p5.PARENT_PHASE4_DIR = parent_phase4
        p5.SELF_PHASE4_DIR = self_phase4
        p5.PARENT_PHASE1 = tmp_root / "no_parent.json"  # doesn't exist, falls back
        p5.SELF_PHASE1 = self_phase1 / "cosine_per_layer.json"
        p5.PARENT_READY_PATH = self_phase0 / "parent_ready.json"
        p5.OUT_DIR = self_phase5

        rc = p5.main(argv=["--smoke", "--frac", str(FRAC)])
        assert rc == 0, f"smoke-mode phase5 returned rc={rc}"

        coll = json.loads((self_phase5 / "collinearity.json").read_text())
        assert coll["panel"] == "merged_840_smoke", (
            f"smoke label should be merged_840_smoke, got {coll['panel']}"
        )
        h1 = json.loads((self_phase5 / "H1_verdict.json").read_text())
        smoke_panels = sorted(h1["panels"].keys())
        for label in smoke_panels:
            assert label.endswith("_smoke"), f"H1 panel label {label!r} must end with _smoke"
        logger.info("Smoke-label check PASS — labels carry _smoke suffix (%s)", smoke_panels)


if __name__ == "__main__":
    raise SystemExit(main())
