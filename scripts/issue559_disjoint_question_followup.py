#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (※, ρ, −, ×, —) in scientific docstrings + labels.
"""Issue #559 follow-up ``disjoint-question-prior`` — VM analysis (plan v4).

The parent run computed the own-response prior from the SAME 20 questions the
trained-side DV uses. This follow-up re-computes it from the 30 committed
Q_test questions (``q50[20:]`` of the pinned #460 ``R_test.json``) the panel
never touches, and asks whether the within-run ranking survives.

Modes (one entrypoint, plan v4 §5.2):

* ``--build-questions`` — derive the 30-question list from HF @ the pinned
  revision (every plan §3 gate re-asserted), write
  ``eval_results/issue_559/disjoint_question_prior/questions_disjoint30.json``
  (committed BEFORE pod launch; the pod consumes it), exit.
* ``--write-stub`` — write full-shape (35×30) stub disjoint inputs into
  ``--out-dir`` for the VM smoke (``is_stub: true``; analysis refuses without
  ``--allow-stub``), exit.
* default — the §4 analysis:

  1. **Entry gate:** recompute the committed rankers (``margin_base``,
     ``min_dist``, 20-q ``prior_margin_own``) through the same code path and
     assert exact reproduction (1e-9, point estimates + per-run ρ maps) vs the
     committed ``within_run_ranking.json`` — the parent's numbers as incumbents.
  2. **Within-run ranking** with {matched-slot, 20-q prior, disjoint-30 prior,
     same-pod repro-20q prior (§14.5 twin), ``min_dist``,
     length-alone-disjoint}: ``i559.within_run_ranking``, dual run/cell
     bootstrap axes, per-statistic RNG re-seeding (v2 §13.2/.4).
  3. **Paired blocks** (§4.4): (a) 20q − disjoint with the inherited ±0.10
     parity band; (b) matched − disjoint; (c) disjoint − length-alone (the
     kill read); plus the §14.5 same-pod twin (repro20q − disjoint).
  4. **Outcome classification** per §6 AS AMENDED by §14.1–3 (straddle =
     PARTIAL/INDETERMINATE; CI-entirely-below-zero = FALSIFIED; both length
     reads co-reported) with §14.7 provenance scoping.
  5. **§14.6 separating reads:** between-persona spread vs the committed
     9.0-nat spread; 15/15 split-half ρ within the 30.
  6. **Sensitivity:** length-residual secondary on the 30 (linear PRIMARY +
     quadratic robustness, the committed ``_ols_residuals`` recipe);
     truncation prior-side-only re-aggregation (§14.4 — NEVER
     ``truncation_matched_block``, whose shared-question premise does not
     exist here); §14.8 count-matched 20-of-30 subsample read; per-persona
     IQR over the 30.
  7. **Figures:** hero ranking strip (parent strip extended with the disjoint
     column), two-prior scatter, paired-difference histograms, length panel.

Outputs ``within_run_ranking_disjoint.json`` to ``--out-dir``. CPU-only; the
smoke is the same script with ``--write-stub`` inputs + ``--allow-stub`` +
reduced ``--n-marginal-boot``.

Usage::

    uv run python scripts/issue559_disjoint_question_followup.py --build-questions
    uv run python scripts/issue559_disjoint_question_followup.py \\
        --out-dir eval_results/issue_559/disjoint_question_prior \\
        --fig-dir figures/issue_559
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue539_residual_per_cohort as i539  # noqa: E402
import issue553_panel as p553  # noqa: E402
import issue559_length_residual_followup as i559len  # noqa: E402
import issue559_panel_analysis as i559  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

SCHEMA_VERSION = "issue559_disjoint_question_followup_v1"
QUESTIONS_SCHEMA = "issue559_disjoint_questions_v1"
PRIOR_SCHEMA = "issue559_base_prior_v1"
REPRO_TOL = 1e-9  # entry-gate exact-reproduction tolerance (deterministic estimates)
PARITY_BAND = i559.PARITY_BAND  # ±0.10, inherited (v2 §11)
HYPOTHESIZED_BAND = (0.55, 0.65)  # reported, never gated (plan v4 §6)
COMMITTED_SPREAD_REF_NATS = 9.0  # committed parent persona spread ([−25.62, −16.65])

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_REV = "a9fc5a9cbc81c4b774ff66da0022f9055e18da5f"  # the pin the pod script uses
R_TEST_HF_PATH = "issue460_marker_at_end/on_policy_R/R_test.json"
DERIVATION_RULE = (
    "q50[20:] — the 30 committed Q_test questions of the pinned #460 R_test.json whose "
    "first 20 ARE the panel's EVAL_QUESTIONS (order-sensitive, verified at plan time + here)"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = p553.common_parser(
        "Issue #559 follow-up — disjoint-question own-response prior (plan v4)"
    )
    parser.set_defaults(
        out_dir=Path("eval_results/issue_559/disjoint_question_prior"),
        fig_dir=Path("figures/issue_559"),
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=Path("eval_results/issue_559/disjoint_question_prior/questions_disjoint30.json"),
        help="committed disjoint-30 question file (written by --build-questions)",
    )
    parser.add_argument(
        "--prior-json",
        type=Path,
        default=Path("eval_results/issue_559/base_prior_own_persona_panel.json"),
        help="committed PARENT 20-question prior (canonical i559.load_prior gates)",
    )
    parser.add_argument(
        "--r-base-own",
        type=Path,
        default=Path("eval_results/issue_559/R_base_own.json"),
        help="committed parent generations (i559.load_prior question cross-check)",
    )
    parser.add_argument(
        "--disjoint-prior-json",
        type=Path,
        default=Path(
            "eval_results/issue_559/disjoint_question_prior/base_prior_own_persona_panel.json"
        ),
        help="NEW disjoint-30 prior from the pod run (load_disjoint_prior gates)",
    )
    parser.add_argument(
        "--disjoint-r-base-own",
        type=Path,
        default=Path("eval_results/issue_559/disjoint_question_prior/R_base_own.json"),
        help="NEW disjoint-30 generations (own_R_token_lens — the length source)",
    )
    parser.add_argument(
        "--repro20q-prior-json",
        type=Path,
        default=Path(
            "eval_results/issue_559/disjoint_question_prior/repro_20q/"
            "base_prior_own_persona_panel.json"
        ),
        help="fresh SAME-POD 20-question prior (entry-gate invocation 1) — the §14.5 "
        "parity twin bounding cross-pod common-mode drift",
    )
    parser.add_argument(
        "--ranking-json",
        type=Path,
        default=Path("eval_results/issue_559/within_run_ranking.json"),
        help="committed parent production ranking — the analysis entry-gate reference",
    )
    parser.add_argument(
        "--length-followup-json",
        type=Path,
        default=Path("eval_results/issue_559/length_residual_followup.json"),
        help="committed length-residual follow-up — source of the +0.3751 length-alone "
        "anchor co-reported per §14.3",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="accept is_stub disjoint inputs (SMOKE ONLY — production refuses stubs)",
    )
    parser.add_argument(
        "--build-questions",
        action="store_true",
        help="derive + write questions_disjoint30.json from HF @ the pin, then exit",
    )
    parser.add_argument(
        "--write-stub",
        action="store_true",
        help="write stub disjoint prior/R JSONs (full 35×30 shape, synthetic values) "
        "into --out-dir and exit — the VM smoke input generator",
    )
    return parser.parse_args(argv)


# ── Question derivation + loading ─────────────────────────────────────────────


def _questions_sha256(questions: list[str]) -> str:
    return hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()


def _derive_disjoint30() -> list[str]:
    """q50[20:] from the pinned #460 R_test.json, with every plan §3 gate."""
    from huggingface_hub import hf_hub_download

    eval_questions = i559._canonical_eval_questions()
    path = hf_hub_download(HF_DATA_REPO, R_TEST_HF_PATH, repo_type="dataset", revision=HF_DATA_REV)
    payload = json.loads(Path(path).read_text())
    assert payload.get("schema_version") == "i460_v1", payload.get("schema_version")
    completions = payload["completions"]
    q_lists = [list(qmap.keys()) for qmap in completions.values()]
    q50 = q_lists[0]
    assert len(completions) == 16 and all(ql == q50 for ql in q_lists), (
        "expected 16 contexts sharing one identical ordered 50-question list"
    )
    assert q50[:20] == eval_questions, "q50[:20] != EVAL_QUESTIONS (order-sensitive)"
    disjoint = q50[20:]
    assert len(disjoint) == 30, len(disjoint)
    assert not set(disjoint) & set(eval_questions), "disjoint-30 overlaps EVAL_QUESTIONS"
    assert not any("※" in q for q in disjoint), "a disjoint question mentions the marker"
    return disjoint


def build_questions(args: argparse.Namespace) -> None:
    """``--build-questions``: write the committed disjoint-30 question file."""
    disjoint = _derive_disjoint30()
    digest = _questions_sha256(disjoint)
    meta = p553.result_metadata(args, "scripts/issue559_disjoint_question_followup.py")
    meta["task"] = 559
    out = {
        "schema_version": QUESTIONS_SCHEMA,
        "n": len(disjoint),
        "questions": disjoint,
        "derivation": {
            "source": R_TEST_HF_PATH,
            "hf_repo": HF_DATA_REPO,
            "hf_revision": HF_DATA_REV,
            "rule": DERIVATION_RULE,
        },
        "sha256_questions": digest,
        "metadata": meta,
    }
    p553.write_json(args.questions_json, out)
    print(f"[build-questions] n={len(disjoint)} sha256={digest}")


def load_questions_file(args: argparse.Namespace) -> list[str]:
    """Load + re-verify the committed disjoint-30 question file."""
    payload = json.loads(args.questions_json.read_text())
    assert payload.get("schema_version") == QUESTIONS_SCHEMA, payload.get("schema_version")
    questions = list(payload["questions"])
    digest = _questions_sha256(questions)
    if digest != payload["sha256_questions"]:
        raise SystemExit(f"questions file sha256 mismatch for {args.questions_json}")
    eval_questions = i559._canonical_eval_questions()
    assert len(questions) == 30, len(questions)
    assert not set(questions) & set(eval_questions), "disjoint-30 overlaps EVAL_QUESTIONS"
    assert not any("※" in q for q in questions), "a disjoint question mentions the marker"
    assert payload["derivation"]["hf_revision"] == HF_DATA_REV, "questions file pin drift"
    return questions


# ── Stub generator (VM smoke input; assert-guarded against production use) ────


def write_stub_disjoint(args: argparse.Namespace) -> None:
    """Write full-shape (35×30) stub disjoint inputs (mirrors i559.write_stub).

    Questions are the CANONICAL disjoint-30 from the committed questions file
    (the ``load_disjoint_prior`` identity gate is never relaxed, stub or not);
    margins are seeded off the parquet's persona-mean ``margin_base`` so the
    ranking / classification paths exercise realistic correlation structure.
    """
    questions = load_questions_file(args)
    df = p553.load_i478_panel(args.i478_parquet)
    personas = sorted(df["held_out_persona"].unique().tolist())
    base_by_p = df.groupby("held_out_persona")["margin_base"].mean()
    rng = np.random.default_rng(559)
    n_q = len(questions)

    per_persona: dict[str, dict] = {}
    R: dict[str, dict[str, str]] = {}
    finish: dict[str, dict[str, str]] = {}
    lens: dict[str, dict[str, int]] = {}
    n_trunc = 0
    for p in personas:
        margins = float(base_by_p[p]) + rng.normal(0.0, 1.0, size=n_q)
        z_eos = 10.0 + rng.normal(0.0, 0.5, size=n_q)
        z_marker = z_eos + margins
        logZ = z_eos + 3.0
        rec = {
            "z_marker_per_q": z_marker.tolist(),
            "z_eos_per_q": z_eos.tolist(),
            "logZ_per_q": logZ.tolist(),
            "logp_marker_per_q": (z_marker - logZ).tolist(),
            "argmax_id_per_q": [151645] * n_q,
            "slot_kind_per_q": ["end_of_response"] * n_q,
            "n_truncated_tokens_per_q": [0] * n_q,
            "finish_reason_per_q": [
                "length" if rng.random() < 0.05 else "stop" for _ in range(n_q)
            ],
            "prior_margin_own": float(np.mean(margins)),
            "prior_margin_own_median": float(np.median(margins)),
            "prior_margin_own_iqr": [
                float(np.percentile(margins, 25)),
                float(np.percentile(margins, 75)),
            ],
            "prior_logp_own": float(np.mean(z_marker - logZ)),
        }
        n_trunc += sum(1 for f in rec["finish_reason_per_q"] if f == "length")
        per_persona[p] = rec
        R[p] = {q: f"stub disjoint response for {p}" for q in questions}
        finish[p] = dict(zip(questions, rec["finish_reason_per_q"], strict=True))
        lens[p] = {q: int(rng.integers(50, 300)) for q in questions}

    n_slots = len(personas) * n_q
    questions_source = {
        "path": str(args.questions_json),
        "hf_repo": HF_DATA_REPO,
        "hf_revision": HF_DATA_REV,
        "rule": DERIVATION_RULE,
        "n": n_q,
        "sha256": _questions_sha256(questions),
    }
    prior_payload = {
        "schema_version": PRIOR_SCHEMA,
        "is_stub": True,
        "eval_questions": questions,
        "personas": personas,
        "per_persona": per_persona,
        "questions_source": questions_source,
        "summary": {
            "n_personas": len(personas),
            "n_questions": n_q,
            "n_slots": n_slots,
            "n_pre_marker_slots": 0,
            "truncation_rate": n_trunc / n_slots,
            "argmax_composition": {
                "marker": {"count": 0, "rate": 0.0},
                "eos": {"count": n_slots, "rate": 1.0},
                "other": {"count": 0, "rate": 0.0},
            },
        },
        "s0_validation_pass": True,
        "metadata": {"note": "STUB — VM smoke input, never production data"},
    }
    r_payload = {
        "schema_version": PRIOR_SCHEMA,
        "is_stub": True,
        "eval_questions": questions,
        "personas": personas,
        "R": R,
        "finish_reasons": finish,
        "own_R_token_lens": lens,
        "truncation_rate": n_trunc / n_slots,
        "questions_source": questions_source,
        "metadata": {"note": "STUB — VM smoke input, never production data"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [
        ("base_prior_own_persona_panel.json", prior_payload),
        ("R_base_own.json", r_payload),
    ]:
        (args.out_dir / name).write_text(json.dumps(obj))
        print(f"[stub] wrote {args.out_dir / name}")


# ── Disjoint-prior loading (i559.load_prior hard-gates the canonical 20) ──────


def load_disjoint_prior(args: argparse.Namespace, questions30: list[str]) -> dict:
    """Load + guard the disjoint prior JSON (plan v4 §4.2).

    Gates: schema ``issue559_base_prior_v1``; stub fence; ``s0_validation_pass``;
    ``eval_questions ==`` the pinned disjoint-30 (order-sensitive, re-derived
    from the committed questions file); 35 × 30 = 1,050 per-q coverage (strict
    — stubs are full-shape by construction); ``questions_source`` sha
    consistency when present; the sibling R file's question identity.
    """
    payload = json.loads(args.disjoint_prior_json.read_text())
    assert payload.get("schema_version") == PRIOR_SCHEMA, payload.get("schema_version")
    if payload.get("is_stub", False) and not args.allow_stub:
        raise SystemExit(
            f"{args.disjoint_prior_json} is a STUB (smoke input) — refusing without --allow-stub"
        )
    if not payload.get("s0_validation_pass", False):
        raise SystemExit("disjoint prior records s0_validation_pass=false — measurement invalid")
    if payload.get("eval_questions") != questions30:
        raise SystemExit(
            "QUESTION-IDENTITY GATE FAILED (disjoint): eval_questions != the pinned "
            "disjoint-30 (order-sensitive) — refusing to read the prior"
        )
    qsrc = payload.get("questions_source")
    if qsrc is not None and qsrc["sha256"] != _questions_sha256(questions30):
        raise SystemExit("questions_source sha256 != committed questions file — provenance drift")
    per_persona = payload.get("per_persona", {})
    bad_lens = {
        p: bad
        for p, rec in per_persona.items()
        if (bad := {k: len(v) for k, v in rec.items() if k.endswith("_per_q") and len(v) != 30})
    }
    if bad_lens:
        raise SystemExit(
            f"COVERAGE GATE FAILED (disjoint): per-q array length != 30 for {sorted(bad_lens)} "
            f"(got {bad_lens}) — partial-question artifact, refusing"
        )
    n_slots = sum(len(rec["logp_marker_per_q"]) for rec in per_persona.values())
    if len(per_persona) != 35 or n_slots != 1_050:
        raise SystemExit(
            "COVERAGE GATE FAILED (disjoint): expected 35 personas × 30 questions = 1,050 "
            f"per-q records, got {len(per_persona)} personas / {n_slots} slots"
        )
    if args.disjoint_r_base_own.exists():
        r_own = json.loads(args.disjoint_r_base_own.read_text())
        if r_own.get("eval_questions") != questions30:
            raise SystemExit(
                "QUESTION-IDENTITY GATE FAILED (disjoint R): eval_questions != the pinned "
                "disjoint-30 — R/prior question drift"
            )
    return payload


# ── Entry gate + classifications ──────────────────────────────────────────────


def assert_entry_gate(ranking: dict, args: argparse.Namespace) -> dict:
    """Recompute the committed rankers; assert exact reproduction (plan v4 §4.1).

    The parent's committed ``within_run_ranking.json`` numbers are the
    incumbents now: ``margin_base``, ``min_dist`` AND the 20-q
    ``prior_margin_own`` must reproduce to 1e-9 (median + full per-run ρ map +
    degenerate count) through this script's join/aggregation before any new
    number is read.
    """
    committed = json.loads(args.ranking_json.read_text())["within_run_ranking"]
    report: dict = {}
    for ranker in ("margin_base", "min_dist", "prior_margin_own"):
        got, want = ranking[ranker], committed[ranker]
        d_med = abs(got["median_rho"] - want["median_rho"])
        if d_med > REPRO_TOL:
            raise SystemExit(
                f"ENTRY GATE FAILED: {ranker} median ρ {got['median_rho']!r} != committed "
                f"{want['median_rho']!r} — join/aggregation drift, refusing to read any "
                "new number"
            )
        if set(got["per_run_rho"]) != set(want["per_run_rho"]):
            raise SystemExit(f"ENTRY GATE FAILED: {ranker} per-run key set drifted")
        worst = max(
            abs(got["per_run_rho"][r] - want["per_run_rho"][r]) for r in want["per_run_rho"]
        )
        if worst > REPRO_TOL:
            raise SystemExit(f"ENTRY GATE FAILED: {ranker} per-run ρ drifted (max {worst})")
        if got["n_degenerate_dropped"] != want["n_degenerate_dropped"]:
            raise SystemExit(f"ENTRY GATE FAILED: {ranker} degenerate-drop count drifted")
        report[ranker] = {
            "median_rho_abs_diff": d_med,
            "per_run_rho_max_abs_diff": worst,
            "pass": True,
        }
        print(f"[entry-gate] {ranker}: median Δ={d_med:.2e}, per-run max Δ={worst:.2e} -> PASS")
    return report


def classify_question_set_parity(run_blk: dict, cell_blk: dict, pair_label: str) -> dict:
    """±0.10 parity classification for a (20q-prior − disjoint-prior) pair (§4.4a).

    Entirely below +0.10 on BOTH axes → question-set-robust at parity;
    entirely above on both → real degradation; anything else → indeterminate
    (a straddle NEVER ships as parity confirmed — v2 §13.1 transposed).
    """
    pr, pc = run_blk["median_diff_ci95"], cell_blk["median_diff_ci95"]
    below = pr["high"] < PARITY_BAND and pc["high"] < PARITY_BAND
    above = pr["low"] > PARITY_BAND and pc["low"] > PARITY_BAND
    if below:
        cls = "QUESTION_SET_ROBUST_AT_PARITY"
        read = (
            f"paired ({pair_label}) median-difference CI entirely below +{PARITY_BAND} on "
            "both resampling axes — question-set-robust at parity"
        )
    elif above:
        cls = "REAL_DEGRADATION"
        read = (
            f"paired ({pair_label}) median-difference CI entirely above +{PARITY_BAND} on "
            "both axes — the 20-question prior is decisively better (real degradation)"
        )
    else:
        cls = "INDETERMINATE"
        read = (
            f"paired ({pair_label}) CI straddles +{PARITY_BAND} (or the axes disagree) — "
            "ships as indeterminate, never as parity confirmed"
        )
    return {
        "classification": cls,
        "read": read,
        "parity_band": PARITY_BAND,
        "ci_run": pr,
        "ci_cell": pc,
    }


def classify_disjoint_outcome(
    prior_blk: dict, len_run: dict, len_cell: dict, committed_length_anchor: float
) -> dict:
    """Plan §6 outcome lattice AS AMENDED by §14.1–3 + §14.7 provenance scoping.

    Exhaustive prior branch (§14.2): CI entirely > 0 on both axes vs not;
    CI entirely below zero = FALSIFIED. Length branch (§14.1): paired
    (disjoint − length-alone-disjoint) CI entirely > 0 → SUCCESS; entirely
    ≤ 0 → affirmative LENGTH_COLLAPSE; straddle → PARTIAL/INDETERMINATE
    ("prior signal robust to question set; superiority over a same-surface
    length proxy not demonstrated at this N") — never narrated as "falls to
    length-alone territory". §14.3: both length reads co-reported; the
    same-batch comparator governs the kill read, the committed +0.3751
    anchor governs interpretation vs the followup-scope's +0.38 prose.
    """
    run_ci = prior_blk["median_ci95_run_boot"]
    cell_ci = prior_blk["median_ci95_cell_boot"]
    prior_pos = run_ci["low"] > 0 and cell_ci["low"] > 0
    prior_neg = run_ci["high"] < 0 and cell_ci["high"] < 0
    lr, lc = len_run["median_diff_ci95"], len_cell["median_diff_ci95"]
    len_pos = lr["low"] > 0 and lc["low"] > 0
    len_collapse = lr["high"] <= 0 and lc["high"] <= 0
    med = prior_blk["median_rho"]
    in_band = HYPOTHESIZED_BAND[0] <= med <= HYPOTHESIZED_BAND[1]

    if prior_neg:
        cls = "FALSIFIED"
        read = (
            "disjoint-prior median ρ CI entirely below zero on both resampling axes — "
            "FALSIFIED (§14.2 exhaustive branch)"
        )
    elif not prior_pos:
        cls = "FALSIFIED"
        read = (
            "disjoint-prior median ρ CI is not entirely > 0 on at least one resampling axis "
            "(conservative dual-axis read) — question-set robustness fails (§6 kill)"
        )
    elif len_pos:
        cls = "SUCCESS"
        read = (
            "disjoint-prior median ρ CI entirely > 0 on both axes AND the paired "
            "(disjoint − length-alone-disjoint) CI entirely > 0 — the prior is robust to "
            "held-out questions from the same committed pool (NOT to arbitrary question "
            "distributions — §14.7 provenance scoping)"
        )
    elif len_collapse:
        cls = "LENGTH_COLLAPSE"
        read = (
            "prior CI entirely > 0 but the paired (disjoint − length-alone-disjoint) CI is "
            "entirely ≤ 0 on both axes — affirmative length-collapse (§14.1): on held-out "
            "questions the prior does not out-rank a same-surface length proxy"
        )
    else:
        cls = "PARTIAL_INDETERMINATE"
        read = (
            "prior signal robust to question set (CI entirely > 0 on both axes); superiority "
            "over a same-surface length proxy not demonstrated at this N (paired CI straddles "
            "0) — §14.1 binding: never narrate as 'falls to length-alone territory'; "
            "affirmative collapse requires the paired CI entirely ≤ 0"
        )
    return {
        "classification": cls,
        "read": read,
        "median_rho": med,
        "in_hypothesized_band_0p55_0p65": bool(in_band),
        "hypothesized_band": list(HYPOTHESIZED_BAND),
        "prior_ci_run": run_ci,
        "prior_ci_cell": cell_ci,
        "paired_disjoint_minus_length_ci_run": lr,
        "paired_disjoint_minus_length_ci_cell": lc,
        "length_reads_co_report": {
            "committed_length_anchor_20q": committed_length_anchor,
            "note": (
                "§14.3: the SAME-BATCH length-alone-disjoint paired comparator governs the "
                "kill read; the committed 20-q +0.3751 anchor governs interpretation vs the "
                "followup-scope's +0.38 prose — they answer different questions"
            ),
        },
    }


# ── Figures ───────────────────────────────────────────────────────────────────


def figure_hero_strip(ranking: dict, fig_dir: Path) -> None:
    """Hero: the parent ranking strip extended with the disjoint-prior column."""
    set_paper_style("blog")
    colors = paper_palette(2)
    order = [
        "prior_margin_own",
        "prior_repro20q",
        "prior_disjoint30",
        "len_oriented_d30",
        "min_dist",
        "margin_base",
    ]
    labels = [
        "own-response prior\n(20 panel questions)",
        "own-response prior\n(fresh same-pod 20-q)",
        "own-response prior\n(30 disjoint questions)",
        "response length alone\n(30 disjoint questions)",
        "distance to nearest\ntrained source",
        "base matched-slot margin\n(needs trained responses)",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    rng = np.random.default_rng(0)
    for xi, ranker in enumerate(order):
        blk = ranking[ranker]
        vals = [v for v in blk["per_run_rho"].values() if not np.isnan(v)]
        color = colors[1] if ranker == "margin_base" else colors[0]
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        ax.plot(np.full(len(vals), xi) + jitter, vals, "o", ms=3.5, alpha=0.45, color=color)
        med = blk["median_rho"]
        ci = blk["median_ci95_run_boot"]
        lo_e = max(0.0, med - ci["low"])  # clamp: CI bounds can sit float-epsilon past median
        hi_e = max(0.0, ci["high"] - med)
        ax.errorbar(
            [xi + 0.30], [med], yerr=[[lo_e], [hi_e]], fmt="o", ms=5, color=color, capsize=3
        )
        ax.plot([xi - 0.22, xi + 0.22], [med, med], color=color, lw=2.4)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Per-run Spearman ρ vs trained EOS margin (35 personas)")
    ax.set_title(
        "Disjoint-question prior — within-run ranking, 80 runs\n"
        "dots = per-run ρ; bar = median; whisker = 95% run-bootstrap CI on the median",
        fontsize=9,
    )
    savefig_paper(fig, "disjoint_question_ranking", dir=fig_dir)
    plt.close(fig)


def figure_two_prior_scatter(prior_20q: dict, prior_d30: dict, fig_dir: Path) -> None:
    """Construction sanity: per-persona 20-q prior vs disjoint-30 prior."""
    set_paper_style("blog")
    colors = paper_palette(1)
    personas = sorted(prior_20q)
    x = [prior_20q[p] for p in personas]
    y = [prior_d30[p] for p in personas]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(x, y, "o", ms=5, alpha=0.7, color=colors[0])
    ax.set_xlabel("own-response prior margin, 20 panel questions (nats)")
    ax.set_ylabel("own-response prior margin, 30 disjoint questions (nats)")
    ax.set_title("Per-persona prior: panel questions vs disjoint questions", fontsize=9)
    savefig_paper(fig, "two_prior_scatter", dir=fig_dir)
    plt.close(fig)


def figure_paired_hists(rho: dict[str, dict[str, float]], fig_dir: Path) -> None:
    """Per-run paired-difference histograms: (20q − disjoint) and (disjoint − length)."""
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    runs = [
        r
        for r in rho["prior_margin_own"]
        if not any(np.isnan(rho[k][r]) for k in ("prior_margin_own", "prior_disjoint30"))
    ]
    d_parity = [rho["prior_margin_own"][r] - rho["prior_disjoint30"][r] for r in runs]
    axes[0].hist(d_parity, bins=24, color=colors[0], alpha=0.8)
    axes[0].axvline(0.0, color="0.4", lw=0.8)
    axes[0].axvline(PARITY_BAND, color=colors[1], lw=1.2, ls="--")
    axes[0].set_xlabel("per-run ρ(20-q prior) − ρ(disjoint prior)")
    axes[0].set_ylabel("runs")
    d_len = [rho["prior_disjoint30"][r] - rho["len_oriented_d30"][r] for r in runs]
    axes[1].hist(d_len, bins=24, color=colors[0], alpha=0.8)
    axes[1].axvline(0.0, color="0.4", lw=0.8)
    axes[1].set_xlabel("per-run ρ(disjoint prior) − ρ(length alone, disjoint)")
    axes[1].set_ylabel("runs")
    fig.suptitle("Paired per-run rank-correlation differences (80 runs)", fontsize=9)
    savefig_paper(fig, "disjoint_paired_hists", dir=fig_dir)
    plt.close(fig)


def figure_length_panel(
    lens_by_p: dict[str, float], prior_d30: dict[str, float], ranking_all: dict, fig_dir: Path
) -> None:
    """Length panel: length vs disjoint prior scatter + residualized-ranking strip."""
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    personas = sorted(prior_d30)
    axes[0].plot(
        [lens_by_p[p] for p in personas],
        [prior_d30[p] for p in personas],
        "o",
        ms=5,
        alpha=0.7,
        color=colors[0],
    )
    axes[0].set_xlabel("mean own-response length over the 30 questions (tokens)")
    axes[0].set_ylabel("disjoint-30 prior margin (nats)")
    order = ["prior_disjoint30", "prior_d30_resid_lin", "len_oriented_d30"]
    labels = ["disjoint prior\n(raw)", "disjoint prior\n(length-residualized)", "length alone"]
    rng = np.random.default_rng(0)
    for xi, ranker in enumerate(order):
        blk = ranking_all[ranker]
        vals = [v for v in blk["per_run_rho"].values() if not np.isnan(v)]
        jitter = (rng.random(len(vals)) - 0.5) * 0.18
        axes[1].plot(
            np.full(len(vals), xi) + jitter, vals, "o", ms=3.5, alpha=0.45, color=colors[0]
        )
        med = blk["median_rho"]
        ci = blk["median_ci95_run_boot"]
        lo_e = max(0.0, med - ci["low"])
        hi_e = max(0.0, ci["high"] - med)
        axes[1].errorbar(
            [xi + 0.30], [med], yerr=[[lo_e], [hi_e]], fmt="o", ms=5, color=colors[0], capsize=3
        )
        axes[1].plot([xi - 0.22, xi + 0.22], [med, med], color=colors[0], lw=2.4)
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("per-run ρ vs trained EOS margin")
    fig.suptitle("Length and the disjoint-question prior", fontsize=9)
    savefig_paper(fig, "disjoint_length_panel", dir=fig_dir)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    if args.build_questions:
        build_questions(args)
        return 0
    if args.write_stub:
        write_stub_disjoint(args)
        return 0

    # ── Inputs + gates ────────────────────────────────────────────────────────
    questions30 = load_questions_file(args)
    parent_payload = i559.load_prior(args)  # canonical 20-q gates (committed parent)
    parent_df, _parent_per_q = i559.prior_frames(parent_payload)

    disjoint_payload = load_disjoint_prior(args, questions30)
    d_df, d_per_q = i559.prior_frames(disjoint_payload)

    # §14.5 same-pod twin: the fresh repro_20q prior through the SAME canonical
    # 20-q loader (namespace shim; no R cross-check file for the twin).
    twin_shim = argparse.Namespace(
        prior_json=args.repro20q_prior_json,
        r_base_own=Path("/nonexistent-no-twin-R-crosscheck"),
        allow_stub=args.allow_stub,
    )
    repro_payload = i559.load_prior(twin_shim)
    repro_df, _ = i559.prior_frames(repro_payload)

    if not args.disjoint_r_base_own.exists():
        raise SystemExit(f"{args.disjoint_r_base_own} missing — the length source is required")
    r_own = json.loads(args.disjoint_r_base_own.read_text())
    lens30_by_p = {
        p: float(np.mean(list(qmap.values()))) for p, qmap in r_own["own_R_token_lens"].items()
    }

    # ── Panel + step-0 gate (identical to production) ─────────────────────────
    df = p553.load_i478_panel(args.i478_parquet)
    p553.step0_i478(df, args.i478_parquet.parent / "summary_logit.json")
    agg = p553.aggregate_run_persona(df)

    agg = agg.merge(
        parent_df[["held_out_persona", "prior_margin_own"]],
        on="held_out_persona",
        how="left",
        validate="many_to_one",
    )
    d_join = d_df.rename(
        columns={
            "prior_margin_own": "prior_disjoint30",
            "prior_margin_own_median": "prior_d30_median_agg",
            "prior_margin_own_trunc_excl": "prior_d30_trunc_excl",
            "prior_half_even": "prior_d30_half_even",
            "prior_half_odd": "prior_d30_half_odd",
        }
    )
    agg = agg.merge(
        d_join[
            [
                "held_out_persona",
                "prior_disjoint30",
                "prior_d30_median_agg",
                "prior_d30_trunc_excl",
                "prior_d30_half_even",
                "prior_d30_half_odd",
            ]
        ],
        on="held_out_persona",
        how="left",
        validate="many_to_one",
    )
    agg = agg.merge(
        repro_df[["held_out_persona", "prior_margin_own"]].rename(
            columns={"prior_margin_own": "prior_repro20q"}
        ),
        on="held_out_persona",
        how="left",
        validate="many_to_one",
    )
    agg["own_len30_mean"] = agg["held_out_persona"].map(lens30_by_p)
    for col in ("prior_margin_own", "prior_disjoint30", "prior_repro20q", "own_len30_mean"):
        assert not agg[col].isna().any(), f"{col} join produced NaN — persona mismatch"
    cell_of_run = dict(zip(agg["run_id"], agg["cell_id"], strict=True))

    # Length-alone baseline on the NEW 30, sign-oriented (parent convention).
    s_len = float(
        np.sign(
            i539._spearman_rho(
                agg["own_len30_mean"].to_numpy(np.float64), agg[i559.DV_COL].to_numpy(np.float64)
            )
        )
    )
    agg["len_oriented_d30"] = s_len * agg["own_len30_mean"]

    # §14.8 count-matched 20-of-30 subsample (seeded, deterministic).
    sub_rng = np.random.default_rng(args.seed)
    sub_idx = sorted(int(i) for i in sub_rng.choice(30, size=20, replace=False))
    sub20_by_p = {p: float(np.mean(margins[sub_idx])) for p, margins in d_per_q.items()}
    agg["prior_d30_sub20"] = agg["held_out_persona"].map(sub20_by_p)

    # Length residualization on the 30 (linear PRIMARY + quadratic robustness).
    y = d_df["prior_margin_own"].to_numpy(np.float64)
    lens = np.array([lens30_by_p[p] for p in d_df["held_out_persona"]], dtype=np.float64)
    lens_z = (lens - lens.mean()) / lens.std()
    resid_lin, fit_lin = i559len._ols_residuals(y, [lens_z])
    resid_quad, fit_quad = i559len._ols_residuals(y, [lens_z, lens_z**2])
    resid_lin_by_p = dict(zip(d_df["held_out_persona"], resid_lin, strict=True))
    resid_quad_by_p = dict(zip(d_df["held_out_persona"], resid_quad, strict=True))
    agg["prior_d30_resid_lin"] = agg["held_out_persona"].map(resid_lin_by_p)
    agg["prior_d30_resid_quad"] = agg["held_out_persona"].map(resid_quad_by_p)
    print(
        f"[resid] disjoint prior ~ length30: linear R²={fit_lin['r2']:.3f}, "
        f"quadratic R²={fit_quad['r2']:.3f}"
    )

    # ── Within-run ranking: headline + sensitivity rankers (one pass) ─────────
    headline = [
        "margin_base",
        "min_dist",
        "prior_margin_own",
        "prior_disjoint30",
        "prior_repro20q",
        "len_oriented_d30",
    ]
    sensitivity_rankers = [
        "prior_d30_resid_lin",
        "prior_d30_resid_quad",
        "prior_d30_sub20",
        "prior_d30_median_agg",
        "prior_d30_half_even",
        "prior_d30_half_odd",
    ]
    print(f"[ranking] within-run ranking ({len(headline) + len(sensitivity_rankers)} rankers) ...")
    ranking_all = i559.within_run_ranking(
        agg, headline + sensitivity_rankers, i559.DV_COL, cell_of_run, args
    )
    ranking = {k: ranking_all[k] for k in headline}

    # ── Analysis entry gate (committed rankers reproduce at 1e-9) ─────────────
    entry_gate = assert_entry_gate(ranking_all, args)

    # ── Paired blocks (§4.4 + §14.5 twin), dual axes ─────────────────────────
    rho = {k: ranking_all[k]["per_run_rho"] for k in [*headline, "prior_d30_resid_lin"]}
    pair_defs = {
        "prior20q_minus_disjoint": ("prior_margin_own", "prior_disjoint30"),
        "matched_minus_disjoint": ("margin_base", "prior_disjoint30"),
        "disjoint_minus_length": ("prior_disjoint30", "len_oriented_d30"),
        "repro20q_minus_disjoint": ("prior_repro20q", "prior_disjoint30"),
        "resid_lin_minus_length": ("prior_d30_resid_lin", "len_oriented_d30"),
    }
    paired: dict = {}
    for name, (a, b) in pair_defs.items():
        paired[f"{name}_run_axis"] = i559.paired_difference_block(rho[a], rho[b], args)
        paired[f"{name}_cell_axis"] = i559.paired_difference_cellaxis(
            rho[a], rho[b], cell_of_run, args
        )

    # ── Classifications ───────────────────────────────────────────────────────
    committed_len_anchor = float(
        json.loads(args.length_followup_json.read_text())["within_run_ranking"]["len_oriented"][
            "median_rho"
        ]
    )
    outcome = classify_disjoint_outcome(
        ranking_all["prior_disjoint30"],
        paired["disjoint_minus_length_run_axis"],
        paired["disjoint_minus_length_cell_axis"],
        committed_len_anchor,
    )
    print(f"[outcome] {outcome['classification']}: {outcome['read']}")
    parity_20q = classify_question_set_parity(
        paired["prior20q_minus_disjoint_run_axis"],
        paired["prior20q_minus_disjoint_cell_axis"],
        "20-q prior − disjoint prior",
    )
    parity_twin = classify_question_set_parity(
        paired["repro20q_minus_disjoint_run_axis"],
        paired["repro20q_minus_disjoint_cell_axis"],
        "same-pod repro-20q prior − disjoint prior",
    )
    parity_sub20 = classify_question_set_parity(
        i559.paired_difference_block(
            rho["prior_margin_own"], ranking_all["prior_d30_sub20"]["per_run_rho"], args
        ),
        i559.paired_difference_cellaxis(
            rho["prior_margin_own"],
            ranking_all["prior_d30_sub20"]["per_run_rho"],
            cell_of_run,
            args,
        ),
        "20-q prior − count-matched 20-of-30 disjoint subsample",
    )
    print(f"[parity 20q-vs-disjoint] {parity_20q['classification']}")
    survival = i559len.classify_survival(
        ranking_all["prior_d30_resid_lin"],
        paired["resid_lin_minus_length_run_axis"],
        paired["resid_lin_minus_length_cell_axis"],
    )

    # ── §14.6 separating reads ────────────────────────────────────────────────
    d_priors = dict(zip(d_df["held_out_persona"], d_df["prior_margin_own"], strict=True))
    p_priors = dict(zip(parent_df["held_out_persona"], parent_df["prior_margin_own"], strict=True))
    spread_d30 = float(max(d_priors.values()) - min(d_priors.values()))
    spread_20q = float(max(p_priors.values()) - min(p_priors.values()))
    split_half_rho = i539._spearman_rho(
        d_df["prior_half_even"].to_numpy(np.float64), d_df["prior_half_odd"].to_numpy(np.float64)
    )
    separating_reads = {
        "between_persona_spread": {
            "disjoint30_nats": spread_d30,
            "parent_20q_nats": spread_20q,
            "committed_reference_nats": COMMITTED_SPREAD_REF_NATS,
            "ratio_disjoint_over_reference": spread_d30 / COMMITTED_SPREAD_REF_NATS,
            "read_rule": "a collapsed spread (≪ the committed 9.0-nat reference) means the "
            "disjoint prior lost its persona signal; a preserved spread with a low ρ-vs-DV "
            "means systematic question-set sensitivity (§14.6a)",
        },
        "split_half_within_30": {
            "spearman_even_vs_odd_15q_halves": split_half_rho,
            "ranking_half_even": ranking_all["prior_d30_half_even"],
            "ranking_half_odd": ranking_all["prior_d30_half_odd"],
            "read_rule": "high split-half reliability + low ρ-vs-DV → systematic; low "
            "split-half → item noise (§14.6b)",
        },
    }

    # ── §14.4 truncation sensitivity: prior-side-only re-aggregation ─────────
    trunc_rate = float(disjoint_payload["summary"]["truncation_rate"])
    trunc_slice = agg[~agg["prior_d30_trunc_excl"].isna()].copy()
    n_personas_dropped = 35 - trunc_slice["held_out_persona"].nunique()
    rank_trunc = i559.within_run_ranking(
        trunc_slice, ["prior_d30_trunc_excl", "prior_disjoint30"], i559.DV_COL, cell_of_run, args
    )
    trunc_paired_run = i559.paired_difference_block(
        rank_trunc["prior_disjoint30"]["per_run_rho"],
        rank_trunc["prior_d30_trunc_excl"]["per_run_rho"],
        args,
    )
    trunc_paired_cell = i559.paired_difference_cellaxis(
        rank_trunc["prior_disjoint30"]["per_run_rho"],
        rank_trunc["prior_d30_trunc_excl"]["per_run_rho"],
        cell_of_run,
        args,
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    personas_sorted = sorted(d_priors)
    two_prior = {
        "pearson_r": float(
            np.corrcoef(
                [p_priors[p] for p in personas_sorted], [d_priors[p] for p in personas_sorted]
            )[0, 1]
        ),
        "spearman_rho": i539._spearman_rho(
            np.array([p_priors[p] for p in personas_sorted]),
            np.array([d_priors[p] for p in personas_sorted]),
        ),
        "n_personas": len(personas_sorted),
    }
    d_view = d_df.set_index("held_out_persona")
    diagnostics = {
        "two_prior_agreement": two_prior,
        "argmax_composition": disjoint_payload["summary"]["argmax_composition"],
        "n_pre_marker_slots": disjoint_payload["summary"]["n_pre_marker_slots"],
        "truncation_rate_global": trunc_rate,
        "parent_truncation_rate_reference": 0.0014,
        "per_persona_iqr_over_30_questions": {
            p: [float(d_view.loc[p, "prior_iqr_low"]), float(d_view.loc[p, "prior_iqr_high"])]
            for p in personas_sorted
        },
        "questions_source": disjoint_payload.get("questions_source"),
        "length_orientation_sign": s_len,
        "length_vs_disjoint_prior": {
            "pearson_r": float(np.corrcoef(lens, y)[0, 1]),
            "spearman_rho": i539._spearman_rho(lens, y),
        },
    }

    # ── Output JSON ───────────────────────────────────────────────────────────
    meta = p553.result_metadata(args, "scripts/issue559_disjoint_question_followup.py")
    meta["task"] = 559
    meta["schema_version"] = SCHEMA_VERSION
    meta["followup_label"] = "disjoint-question-prior"
    meta["disjoint_prior_is_stub"] = bool(disjoint_payload.get("is_stub", False))
    meta["questions_sha256"] = _questions_sha256(questions30)
    out = {
        "metadata": meta,
        "dv": i559.DV_COL,
        "entry_gate": entry_gate,
        "within_run_ranking": ranking,
        "paired": paired,
        "outcome_classification": outcome,
        "question_set_parity_vs_20q": parity_20q,
        "question_set_parity_same_pod_twin": parity_twin,
        "separating_reads": separating_reads,
        "sensitivity": {
            "length_residual": {
                "residualization": {"linear": fit_lin, "quadratic": fit_quad},
                "ranking_resid_lin": ranking_all["prior_d30_resid_lin"],
                "ranking_resid_quad": ranking_all["prior_d30_resid_quad"],
                "survival_verdict": survival,
            },
            "truncation_prior_side_only": {
                "method": "§14.4 — prior re-aggregated over non-truncated questions only; DV "
                "stays the all-question aggregate (truncation_matched_block's shared-question "
                "premise does not exist for the disjoint 30)",
                "truncation_rate": trunc_rate,
                "n_personas_dropped": int(n_personas_dropped),
                "ranking": rank_trunc,
                "paired_full_minus_trunc_excl_run_axis": trunc_paired_run,
                "paired_full_minus_trunc_excl_cell_axis": trunc_paired_cell,
            },
            "count_matched_subsample_20_of_30": {
                "subsample_question_indices": sub_idx,
                "ranking": ranking_all["prior_d30_sub20"],
                "parity_vs_20q": parity_sub20,
                "note": "§14.8 — separates question IDENTITY from question COUNT in the "
                "parity read (30-q prior is slightly less noisy by construction)",
            },
            "median_aggregated_prior": ranking_all["prior_d30_median_agg"],
        },
        "diagnostics": diagnostics,
        "per_persona": {
            p: {
                "prior_margin_own_20q": p_priors[p],
                "prior_disjoint30": d_priors[p],
                "prior_repro20q": float(
                    repro_df.set_index("held_out_persona").loc[p, "prior_margin_own"]
                ),
                "own_len30_mean": lens30_by_p[p],
                "prior_d30_resid_lin": float(resid_lin_by_p[p]),
            }
            for p in personas_sorted
        },
    }
    p553.write_json(args.out_dir / "within_run_ranking_disjoint.json", out)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("[figures] hero strip + scatter + paired hists + length panel ...")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    figure_hero_strip(ranking_all, args.fig_dir)
    figure_two_prior_scatter(p_priors, d_priors, args.fig_dir)
    figure_paired_hists(rho, args.fig_dir)
    figure_length_panel(lens30_by_p, d_priors, ranking_all, args.fig_dir)

    print("[done] disjoint-question follow-up analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
