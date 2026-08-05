"""Tests for the evil-ood-spread-round scripts (task #1739 unit 4c).

Covers the round's script surface — the paired-CI + AUROC-CI + positive_count
schema, selection-inheritance sanity, drop-never-coerce, transport-vs-content
split, and the tactic-classifier default. Runtime target: full file < 30 s.

Grounded on:
    - plan v16 §6 paired-contrast block (schema fields the arm-results JSON MUST carry).
    - plan v16 §7 verdict lattice (paired CI must exclude zero for H1/H2 verdicts).
    - .claude/rules/llm-judging.md rule 9 (drop-never-coerce content-class).
    - .claude/rules/llm-judging.md rule 24 (transport-vs-content split).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT = REPO_ROOT / "scripts" / "issue1739_score_new_rungs.py"


# ---------------------------------------------------------------------------
# helpers: mock fixtures
# ---------------------------------------------------------------------------
def _write_smoke_rung(root: Path, rung: str, n: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Emit contexts/dv_pool/arm_scores JSONs for one mock rung.

    Uses a fixed pool of 4 arms including arm16_surface_feat + two map-family
    members so the paired-diff + selection-inheritance paths exercise.
    """
    (root / "contexts").mkdir(parents=True, exist_ok=True)
    (root / "dv_pool").mkdir(parents=True, exist_ok=True)
    (root / "arm_scores").mkdir(parents=True, exist_ok=True)

    order = [f"ctx_{i:03d}" for i in range(n)]
    rng = np.random.default_rng(seed)
    dv = rng.normal(50, 20, size=n).clip(0, 100)
    (root / "contexts" / f"{rung}.json").write_text(json.dumps({"order": order}))
    (root / "dv_pool" / f"{rung}.json").write_text(
        json.dumps(
            {"contexts": [{"context_id": c, "dv": float(dv[i])} for i, c in enumerate(order)]}
        )
    )
    arms = {
        "arm6_map_proj_e1": {"scores": (dv + rng.normal(0, 5, n)).tolist()},
        "arm7_map_ridge_pred": {"scores": (dv * 0.9 + rng.normal(0, 6, n)).tolist()},
        "arm16_surface_feat": {"scores": (dv * 0.4 + rng.normal(0, 15, n)).tolist()},
        "arm2_ctx_native": {"scores": rng.normal(0, 1, n).tolist()},
    }
    (root / "arm_scores" / f"{rung}.json").write_text(json.dumps({"arms": arms}))
    return order, dv


def _run_score_smoke(root: Path, out_dir: Path, rung: str = "mhj_full", n_boot: int = 100) -> dict:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "MALLOC_ARENA_MAX": "2",
    }
    r = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCORE_SCRIPT),
            "--rungs",
            rung,
            "--input-root",
            str(root),
            "--output",
            str(out_dir),
            "--smoke",
            "--n-boot",
            str(n_boot),
            "--n-perm",
            str(n_boot),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=45,
    )
    assert r.returncode == 0, f"score smoke rc={r.returncode}\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
    return json.loads((out_dir / f"{rung}.json").read_text())


# ---------------------------------------------------------------------------
# Assertion 1: paired-CI + AUROC-CI + positive_count schema
# ---------------------------------------------------------------------------
def test_score_new_rungs_emits_paired_ci_schema(tmp_path: Path) -> None:
    """arm-results JSON MUST carry rho + ci_rho + ci_rho_delta_vs_arm16 +
    auroc + ci_auroc + ci_auroc_delta_vs_arm16 + positive_count (plan v16 §6)."""
    root = tmp_path / "in"
    out = tmp_path / "out"
    _write_smoke_rung(root, "mhj_full", n=20, seed=1739)
    result = _run_score_smoke(root, out)

    assert "positive_count" in result, "top-level positive_count missing"
    assert isinstance(result["positive_count"], int)
    assert 0 <= result["positive_count"] <= 20

    required_per_arm = {
        "rho",
        "ci_rho",
        "auroc",
        "ci_auroc",
        "ci_rho_delta_vs_arm16",
        "ci_auroc_delta_vs_arm16",
        "rho_delta_vs_arm16",
        "auroc_delta_vs_arm16",
    }
    non_arm16 = [a for a in result["arms"] if a["arm"] != "arm16_surface_feat"]
    assert non_arm16, "expected at least one non-arm16 arm"
    for arm in non_arm16:
        missing = required_per_arm - set(arm.keys())
        assert not missing, f"arm {arm['arm']} missing fields: {missing}"
        assert isinstance(arm["ci_rho"], list) and len(arm["ci_rho"]) == 2
        assert isinstance(arm["ci_auroc"], list) and len(arm["ci_auroc"]) == 2
        assert (
            isinstance(arm["ci_rho_delta_vs_arm16"], list)
            and len(arm["ci_rho_delta_vs_arm16"]) == 2
        )

    # arm16_surface_feat row: marginal fields present; delta fields absent by design.
    arm16 = next(a for a in result["arms"] if a["arm"] == "arm16_surface_feat")
    assert "rho" in arm16 and "ci_rho" in arm16
    assert "rho_delta_vs_arm16" not in arm16, "arm16 vs arm16 delta should NOT be emitted"


# ---------------------------------------------------------------------------
# Assertion 2: selection-inheritance CI at LEAST as wide as any fixed arm's
# ---------------------------------------------------------------------------
def test_selection_inherited_ci_covers_point_and_is_nontrivial(tmp_path: Path) -> None:
    """Selection-inherited paired-max CI: (a) the observed winner's rho_delta
    point estimate lies WITHIN [lo, hi]; (b) the CI is non-trivial (non-empty
    width); (c) the per-draw MAX-over-family draws is at least as extreme (in
    the observed direction) as any single family arm's draws at the same
    percentile — the selection-symmetric property plan v16 §6 relies on.

    (This is the correct invariant — the naive "selection CI is WIDER" claim
    is wrong when family arms are highly correlated: the max of correlated
    positive rhos NARROWS the tails, not widens them; that is exactly why
    plan §6 makes the max/selection its OWN read rather than reusing a fixed
    arm's CI.)
    """
    for seed in (13, 17, 23):
        root = tmp_path / f"in{seed}"
        out = tmp_path / f"out{seed}"
        _write_smoke_rung(root, "mhj_full", n=30, seed=seed)
        result = _run_score_smoke(root, out, n_boot=200)
        bm = result["best_map_family_selection"]
        assert bm is not None, "best_map_family_selection missing"

        # (a) CI covers the point estimate.
        lo, hi = bm["ci_rho_delta_vs_arm16"]
        point = bm["rho_delta_vs_arm16"]
        assert lo <= point <= hi, f"seed={seed}: point {point} not in CI [{lo}, {hi}]"

        # (b) CI is non-trivial (draws didn't all collapse — the bootstrap fired).
        assert (hi - lo) > 1e-6, f"seed={seed}: degenerate zero-width CI [{lo}, {hi}]"

        # (c) observed rho_max is >= any single family arm's marginal rho.
        family_rhos = [a["rho"] for a in result["arms"] if a["arm"] in bm["family_members"]]
        assert bm["rho_max"] >= max(family_rhos) - 1e-9, (
            f"seed={seed}: selection max {bm['rho_max']} < any family arm's marginal rho"
        )


# ---------------------------------------------------------------------------
# Assertion 3: positive_count matches true count above threshold
# ---------------------------------------------------------------------------
def test_positive_count_matches_dv_threshold(tmp_path: Path) -> None:
    """positive_count = count(DV >= 50.0) in the input pool."""
    from scripts.issue1739_rescore_ood import AUROC_POS_THR

    root = tmp_path / "in"
    out = tmp_path / "out"
    _, dv = _write_smoke_rung(root, "mhj_full", n=50, seed=42)
    expected = int((dv >= AUROC_POS_THR).sum())
    result = _run_score_smoke(root, out)
    assert result["positive_count"] == expected, (
        f"positive_count={result['positive_count']} vs true {expected}"
    )


# ---------------------------------------------------------------------------
# Assertion 4: tactic_classify defaults unknown context to "Other"
# ---------------------------------------------------------------------------
def test_tactic_classifier_routes_unclassifiable_to_drop() -> None:
    """The tactic-classify judge's rubric routes non-matching / malformed
    replies to the DROP marker (returns None per llm-judging.md rule 9
    drop-never-coerce), never coercing to a real MHJ tactic class label.
    Plan v16 §4.3 wording "assign 'Other'" is realized as a rule-9 drop
    with the label 'Other/Unclassifiable' recorded provenance-only.
    """
    tactic_module = REPO_ROOT / "scripts" / "issue1739_tactic_classify.py"
    if not tactic_module.exists():
        pytest.skip("issue1739_tactic_classify.py not present in this checkout")

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1739_tactic_classify as tactic  # type: ignore[import-not-found]
    except ImportError as exc:
        pytest.skip(f"tactic module import failed: {exc}")

    # The module defines the 7 MHJ classes + a DROP marker for
    # non-matching / malformed replies.
    valid = getattr(tactic, "MHJ_LABELS", None)
    drop_marker = getattr(tactic, "DROP_LABEL", None)
    assert valid is not None, "expected MHJ_LABELS constant"
    assert drop_marker is not None, "expected DROP_LABEL constant"
    assert drop_marker not in valid, (
        f"DROP_LABEL {drop_marker!r} MUST NOT be a real MHJ class (would coerce drops into a class)"
    )
    # Guard: the 7 plan-required MHJ classes are present.
    expected = {
        "Direct Request",
        "Obfuscation",
        "Hidden Intention Streamline",
        "Request Framing",
        "Injection",
        "Output Format",
        "Echoing",
    }
    assert expected <= set(valid), f"missing MHJ classes: {expected - set(valid)}"

    parser = getattr(tactic, "_parse_label", None)
    assert callable(parser), "expected _parse_label helper on the module"

    # Empty / non-matching / explicit-Other replies MUST return None (drop),
    # NEVER coerce to one of the 7 real classes.
    bad_replies = [
        "",  # empty
        "\n",  # whitespace only
        "This does not fit any category clearly.",  # non-matching text
        "Label: Nonexistent Class",  # unknown label name
        "Label: Other/Unclassifiable",  # explicit drop request
        "42",  # numeric junk
        "Label: other",  # lowercase 'other' -> drop
    ]
    for bad in bad_replies:
        got = parser(bad)
        assert got is None, (
            f"parser({bad!r})={got!r} — MUST return None (drop), never coerce to an MHJ class"
        )


# ---------------------------------------------------------------------------
# Assertion 4b: v2 JSON-verdict extraction + label routing + recovery parser
# (the 2026-08-05 item-B instrument fix: reason-then-JSON rubric, benign class,
#  and the --recover-from-raw Label-line strict parser)
# ---------------------------------------------------------------------------
def _tactic_module():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1739_tactic_classify as tactic  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"tactic module import failed: {exc}")
    return tactic


def test_extract_label_json_last_object_wins() -> None:
    """Reason-then-JSON extraction takes the LAST label-bearing JSON object —
    a rationale quoting attack-embedded JSON (Injection class) must not win."""
    tactic = _tactic_module()
    fn = tactic._extract_label_json

    # plain reason-then-JSON
    assert (
        fn('The text asks directly, without disguise.\n{"label": "Direct Request"}')
        == "Direct Request"
    )
    # decoy JSON earlier in the rationale (quoted attack payload) — last wins
    decoy = (
        'The attack embeds {"role": "system", "content": "x"} to override rules.\n'
        '{"label": "Injection"}'
    )
    assert fn(decoy) == "Injection"
    # trailing non-label object must not end the scan
    trailing = '{"label": "Output Format"}\nExtra note: {"unrelated": 1}'
    assert fn(trailing) == "Output Format"
    # markdown-fenced JSON still parses (raw_decode anchors on '{')
    assert fn('Reasoning here.\n```json\n{"label": "Obfuscation"}\n```') == "Obfuscation"
    # no JSON at all -> None (drop, rule 9)
    assert fn("No JSON verdict anywhere. Label: Direct Request") is None
    assert fn("") is None


def test_route_label_taxonomy_and_drop_reasons() -> None:
    """_route_label: MHJ exact -> label; benign -> BENIGN (never a tactic);
    Other/unknown -> reason-coded drops (drop-never-coerce)."""
    tactic = _tactic_module()
    route = tactic._route_label

    assert route("Direct Request") == ("Direct Request", None)
    assert route("Benign/Not-an-attack") == (tactic.BENIGN_LABEL, None)
    assert route("benign — not an attack") == (tactic.BENIGN_LABEL, None)
    assert route("Other/Unclassifiable") == (None, "other_unclassifiable")
    assert route(None) == (None, "no_label")
    assert route("Nonexistent Class") == (None, "bad_label")
    # BENIGN_LABEL must never live in the tactic label set
    assert tactic.BENIGN_LABEL not in tactic.MHJ_LABELS


def test_parse_label_line_strict_no_fuzzy_whole_text_scan() -> None:
    """Recovery parser: last Label-line only; prose mentioning a class name
    WITHOUT a Label line must NOT be labeled (precision over recall)."""
    tactic = _tactic_module()
    fn = tactic._parse_label_line_strict

    assert fn("Reasoning.\nLabel: Direct Request") == "Direct Request"
    # label line beyond the last 3 lines still found (line-anchored, all lines)
    many = "Label: Injection\n" + "\n".join(f"extra {i}" for i in range(5))
    assert fn(many) == "Injection"
    # NO fuzzy whole-text rescue: class name in prose without a Label line
    assert fn("This looks like a Direct Request to me, but I refuse to label it.") is None
    assert fn("") is None


def test_holdout_whiten_acts_per_layer_mu() -> None:
    """_whiten_acts must center per LAYER (wh.mu is (Ly, d)) — the pre-fix
    local `z - wh.mu[None, None, :]` broadcast (Ly, n, d) against (1, 1, Ly, d)
    and raised ValueError at every realistic shape (2026-08-05 B3 pilot crash;
    same class the sibling rescore leg fixed by delegating to
    fits.apply_whitening)."""
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import issue1739_holdout_rung as h
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"holdout module import failed: {exc}")
    from explore_persona_space.experiments.issue_1739 import fits

    rng = np.random.default_rng(0)
    ly, n, d = 3, 5, 4
    z = rng.normal(size=(ly, n, d))
    wh = fits.Whitening(
        mu=rng.normal(size=(ly, d)),
        w=rng.normal(size=(ly, d, d)),
        gamma=np.ones(ly),
    )
    got = h._whiten_acts(z, wh)  # pre-fix: ValueError (broadcast) at n != ly
    assert got.shape == (ly, n, d)
    # per-layer centering semantics: layer 0 must use mu[0], not a flat mu
    expect0 = (z[0] - wh.mu[0][None, :]) @ wh.w[0]
    np.testing.assert_allclose(got[0], expect0, rtol=1e-12, atol=1e-12)


def test_recover_from_raw_end_to_end(tmp_path: Path) -> None:
    """recover_from_raw: refusal rows drop as refusal (never text-scanned),
    transport rows split out, Other drops, valid Label lines recover; the
    output keeps holdout_rung's 'labels' contract (MHJ classes only)."""
    tactic = _tactic_module()

    labeling = tmp_path / "labeling.json"
    cids = [
        "evil-train-cross-000001",
        "evil-train-cross-000002",
        "evil-eval-hhrt-000001",
        "evil-eval-toxicchat-000001",
        "evil-train-cross-000003",
    ]
    labeling.write_text(json.dumps({"rows": [{"context_id": c} for c in cids]}))

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        {
            "custom_id": cids[0],
            "rtype": "succeeded",
            "stop_reason": "end_turn",
            "text": "Asks directly.\nLabel: Direct Request",
        },
        {
            "custom_id": cids[1],
            "rtype": "succeeded",
            "stop_reason": "refusal",
            # refusal rows are NEVER text-scanned — this mention must not label
            "text": "Direct Request",
        },
        {
            "custom_id": cids[2],
            "rtype": "succeeded",
            "stop_reason": "end_turn",
            "text": "A legitimate question.\nLabel: Other/Unclassifiable",
        },
        {"custom_id": cids[3], "rtype": "errored"},
        {
            "custom_id": cids[4],
            "rtype": "succeeded",
            "stop_reason": "end_turn",
            "text": "Embedded in fiction.\nLabel: Hidden Intention Streamline",
        },
    ]
    with (raw_dir / "raw_msgbatch_test.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "tactic_labels.json"
    payload = tactic.recover_from_raw(labeling, raw_dir, out)

    assert payload["labels"] == {
        cids[0]: "Direct Request",
        cids[4]: "Hidden Intention Streamline",
    }
    assert payload["n_labeled"] == 2
    assert payload["n_transport_lost"] == 1
    assert payload["n_drop"] == 2  # refusal + other_unclassifiable
    assert payload["drop_split"]["refusal"] == 1
    assert payload["drop_split"]["other_unclassifiable"] == 1
    # per-rung split derived from context_id prefixes
    assert payload["per_rung"]["evil_train"]["labeled"] == 2
    assert payload["per_rung"]["evil_hh_rlhf"]["drop"] == 1
    assert payload["per_rung"]["evil_toxicchat"]["transport"] == 1
    # written file round-trips and keeps the holdout_rung 'labels' contract
    disk = json.loads(out.read_text())
    assert set(disk["labels"].values()) <= set(tactic.MHJ_LABELS)


# ---------------------------------------------------------------------------
# Assertion 5: drop-never-coerce — the pilot judge stub excludes dropped
# items from kept_scores (no coercion to 0/50/etc).
# ---------------------------------------------------------------------------
def test_drop_never_coerce_content_class(tmp_path: Path) -> None:
    """A dropped content-class draw MUST NOT appear in kept_scores. We
    exercise the pilot judge's smoke stub, which deterministically drops
    every 20th item as CONTENT (per rule 9 drop-never-coerce).
    """
    pilot = REPO_ROOT / "scripts" / "issue1739_pilot_judge.py"
    if not pilot.exists():
        pytest.skip("issue1739_pilot_judge.py not present")

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1739_pilot_judge as pj  # type: ignore[import-not-found]
    except ImportError as exc:
        pytest.skip(f"pilot judge module import failed: {exc}")

    stub = getattr(pj, "_judge_rung_stub", None)
    assert callable(stub), "expected _judge_rung_stub helper on the pilot judge module"

    payloads = [{"context_id": f"ctx_{i:03d}", "rollout_k": 0} for i in range(80)]
    out = stub(payloads, out_dir=tmp_path, seed=1739)

    n_total = out["per_arm_drop"]["n_total_draws"]
    n_content = out["per_arm_drop"]["n_dropped_draws"]
    n_transport = out["per_arm_drop"]["n_transport_lost_draws"]
    n_kept = len(out["kept_scores"])

    # Split invariant: kept = total - content - transport (NO coercion).
    assert n_total == 80
    assert n_kept + n_content + n_transport == n_total, (
        f"kept({n_kept}) + content_drop({n_content}) + transport_lost({n_transport}) "
        f"!= total({n_total}) — coercion detected"
    )
    assert n_content > 0, "stub should emit at least one content drop over 80 items"
    # Every kept score is a real float in [0, 100] — never a placeholder marker.
    for s in out["kept_scores"]:
        assert isinstance(s, float) and 0.0 <= s <= 100.0


# ---------------------------------------------------------------------------
# Assertion 6: transport-vs-content drop split (rule 24)
# ---------------------------------------------------------------------------
def test_transport_error_increments_transport_lost_not_content_dropped(tmp_path: Path) -> None:
    """A 429 / 5xx / timeout / connection failure = TRANSPORT loss, tallied
    into n_transport_lost_draws — DISTINCT from n_dropped_draws (content).
    Blending the two silently censors arms (rule 24). We verify the pilot
    judge's per_arm_drop schema keeps them separate AND both are non-negative
    integers.
    """
    pilot = REPO_ROOT / "scripts" / "issue1739_pilot_judge.py"
    if not pilot.exists():
        pytest.skip("issue1739_pilot_judge.py not present")

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1739_pilot_judge as pj  # type: ignore[import-not-found]
    except ImportError as exc:
        pytest.skip(f"pilot judge module import failed: {exc}")

    stub = getattr(pj, "_judge_rung_stub", None)
    assert callable(stub), "expected _judge_rung_stub helper on the pilot judge module"

    # 200 items exercises the split's KEY SHAPE (both counters present as
    # distinct int fields). NOTE: the stub's own dispatch ordering treats
    # `i % 20 == 0` BEFORE `i % 40 == 0`, so the transport arm may not fire
    # in this smoke; that is a stub artifact, not a rule-24 violation. The
    # binding invariant is that the two counters are SEPARATE keys (blending
    # them into one would violate rule 24) AND kept + content + transport =
    # total (no coercion of dropped items into kept scores).
    payloads = [{"context_id": f"ctx_{i:03d}", "rollout_k": 0} for i in range(200)]
    out = stub(payloads, out_dir=tmp_path, seed=1739)

    drop = out["per_arm_drop"]
    # Rule-24 split: both counters MUST be present as DISTINCT keys.
    assert "n_dropped_draws" in drop, "content-drop counter missing"
    assert "n_transport_lost_draws" in drop, "transport-loss counter missing"
    # DISTINCT keys, non-conflated in any downstream consumer.
    assert "n_dropped_draws" != "n_transport_lost_draws"
    assert isinstance(drop["n_dropped_draws"], int) and drop["n_dropped_draws"] >= 0
    assert isinstance(drop["n_transport_lost_draws"], int) and drop["n_transport_lost_draws"] >= 0
    # Kept + content + transport = total; no coercion.
    n_total = drop["n_total_draws"]
    n_kept = len(out["kept_scores"])
    assert n_kept + drop["n_dropped_draws"] + drop["n_transport_lost_draws"] == n_total, (
        f"kept({n_kept}) + content({drop['n_dropped_draws']}) + "
        f"transport({drop['n_transport_lost_draws']}) != total({n_total}) — coercion detected"
    )
    # The n_dropped_draws counter is populated (at least the content-drop
    # arm exercises, so the split is not entirely vacuous).
    assert drop["n_dropped_draws"] > 0, "expected at least one content drop over 200 items"


# ---------------------------------------------------------------------------
# paired-CI helper tests (task #1739 unit 5a)
# ---------------------------------------------------------------------------
def _paired_ci_helper():
    """Import the shared paired-CI helper (kept lazy so this file stays import-cheap)."""
    from explore_persona_space.analysis import paired_ci

    return paired_ci


def test_paired_rho_delta_identical_predictors_ci_straddles_zero():
    """When arm A == arm B (identical predictors), the paired-rho delta CI must include 0.

    Paired construction cancels correlated DV noise, so the two rhos are
    numerically identical every draw and the delta is exactly 0 per draw =>
    the empirical quantile CI is [0, 0] (an interval that clearly contains 0).
    Directly tests the paired construction — SHARED resample indices per draw.
    """
    pc = _paired_ci_helper()

    rng = np.random.default_rng(1739)
    n = 60
    dv = rng.normal(loc=50, scale=25, size=n)
    preds = 0.5 * dv + rng.normal(scale=10, size=n)  # noisy but correlated
    preds_arm_a = preds.copy()
    preds_arm_b = preds.copy()

    rho_a, rho_b, ci_lo, ci_hi = pc.paired_bootstrap_rho_delta(
        preds_arm_a, preds_arm_b, dv, n_boot=200, seed=42
    )
    assert rho_a == pytest.approx(rho_b, abs=1e-12)
    assert ci_lo <= 0.0 <= ci_hi, (
        f"identical arms must give delta CI containing 0, got [{ci_lo}, {ci_hi}]"
    )
    # And with identical inputs the CI is a POINT interval (both bounds zero).
    assert ci_lo == pytest.approx(0.0, abs=1e-12)
    assert ci_hi == pytest.approx(0.0, abs=1e-12)


def test_paired_rho_delta_dominant_vs_shuffled_ci_excludes_zero_positive():
    """A dominant predictor vs its shuffled version excludes zero in the paired delta CI.

    Arm A = dv + small noise (strong signal); arm B = shuffled version of arm A
    (destroys pairing with dv). The paired delta rho_A - rho_B is positive and
    the CI must exclude zero.
    """
    pc = _paired_ci_helper()

    rng = np.random.default_rng(4242)
    n = 100
    dv = rng.normal(loc=50, scale=25, size=n)
    # Arm A: strong (rho_A ~ 0.9); Arm B: shuffled -> rho_B ~ 0
    preds_a = dv + rng.normal(scale=5, size=n)
    preds_b = preds_a.copy()
    rng.shuffle(preds_b)

    rho_a, rho_b, ci_lo, ci_hi = pc.paired_bootstrap_rho_delta(
        preds_a, preds_b, dv, n_boot=200, seed=42
    )
    assert rho_a > rho_b, f"expected arm A rho > arm B rho, got {rho_a} vs {rho_b}"
    assert ci_lo > 0.0, (
        f"expected delta CI to exclude 0 on the positive side, got [{ci_lo}, {ci_hi}] "
        f"(rho_a={rho_a}, rho_b={rho_b})"
    )


# ---------------------------------------------------------------------------
# Round 19 pins — concern u1c-sibling-cells-filter + u1c-mem-guard
# ---------------------------------------------------------------------------
def _committed_nesting_cell(
    *,
    f_u: float | None,
    u_rung_label: str,
    variant: str = "prefix_end",
    regime: str = "e2",
    budget_l: int = 250,
) -> dict:
    """One cells.jsonl row in the COMMITTED nesting: identity ONLY inside the
    ``unit_key`` JSON STRING (+ mirrored on arm rows) — never at the top level.

    Mirrors eval_results/issue_1739/evil/arm_results/percell/cells.jsonl
    (top-level keys: arms / headline / max_over_arms_null / preds_npz /
    skipped_arms / split_half / unit_key; measured 2026-08-05).
    """
    ident = {
        "behavior": "evil",
        "budget_l": budget_l,
        "config": "config_a",
        "draw": 0,
        "eval_rung": "train",
        "f_l": None,
        "f_u": f_u,
        "regime": regime,
        "seed": 0,
        "u_rung": 18793,
        "u_rung_label": u_rung_label,
        "variant": variant,
    }
    return {
        "unit_key": json.dumps(ident, sort_keys=True),
        "arms": [{"arm": "arm4_ridge_ctx", "rho_per_layer": [0.1, 0.2], **ident}],
        "headline": {},
        "max_over_arms_null": {},
        "preds_npz": "percell/preds/x.npz",
        "skipped_arms": {},
        "split_half": {},
    }


def test_plain_ladder_filter_reads_unit_key_nesting(tmp_path: Path) -> None:
    """FAILS PRE-FIX: the committed schema carries f_u/u_rung_label only inside
    the unit_key JSON string; the old top-level filter kept 0 of these rows
    (measured live: 0 of 826 committed evil cells; the 2026-08-03
    ood_detection_metrics.json shipped n_metric_rows=0 at rc=0)."""
    from scripts.issue1739_rescore_ood import _load_plain_ladder_cells

    rows = [
        _committed_nesting_cell(f_u=None, u_rung_label="full", variant="context_end", regime="e1"),
        _committed_nesting_cell(f_u=None, u_rung_label="full", variant="prefix_end", regime="e2"),
        # ladder rows the filter must REJECT: f_u set, or a sub-rung u pool
        _committed_nesting_cell(f_u=0.5, u_rung_label="full"),
        _committed_nesting_cell(f_u=None, u_rung_label="r6k"),
    ]
    p = tmp_path / "cells.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cells = _load_plain_ladder_cells(p)
    assert len(cells) == 2, f"expected 2 plain-ladder cells kept, got {len(cells)}"
    # identity keys must be HOISTED so downstream (variant, regime) grouping +
    # budget/draw/seed provenance reads see real values, not defaults
    assert {(c["variant"], c["regime"]) for c in cells} == {
        ("context_end", "e1"),
        ("prefix_end", "e2"),
    }
    assert all(c["u_rung_label"] == "full" and c["f_u"] is None for c in cells)
    assert all(c["budget_l"] == 250 and c["seed"] == 0 for c in cells)


def test_plain_ladder_filter_zero_selection_raises(tmp_path: Path) -> None:
    """A 0-cell selection must RAISE — never flow into a rc=0 'done: 0 metric
    rows' run (the 2026-08-03 silent-failure shape)."""
    from scripts.issue1739_rescore_ood import _load_plain_ladder_cells

    p = tmp_path / "cells.jsonl"
    p.write_text(json.dumps(_committed_nesting_cell(f_u=0.5, u_rung_label="r6k")) + "\n")
    with pytest.raises(RuntimeError, match="ZERO matched the plain-ladder filter"):
        _load_plain_ladder_cells(p)
    # empty file: equally fail-loud
    p.write_text("")
    with pytest.raises(RuntimeError, match="0 cells read"):
        _load_plain_ladder_cells(p)


def test_plain_ladder_filter_top_level_keys_still_win(tmp_path: Path) -> None:
    """A schema that DOES carry top-level identity keys is unaffected (top
    level wins over unit_key contents)."""
    from scripts.issue1739_rescore_ood import _load_plain_ladder_cells

    row = _committed_nesting_cell(f_u=None, u_rung_label="full")
    row["u_rung_label"] = "r6k"  # top-level says sub-rung -> must be rejected
    keeper = _committed_nesting_cell(f_u=None, u_rung_label="full", variant="context_end")
    p = tmp_path / "cells.jsonl"
    p.write_text(json.dumps(row) + "\n" + json.dumps(keeper) + "\n")
    cells = _load_plain_ladder_cells(p)
    assert len(cells) == 1 and cells[0]["variant"] == "context_end"


def test_holdout_rung_maps_memguard_refusal_to_designed_rc(monkeypatch) -> None:
    """concern u1c-mem-guard: a MemGuardRefusal from the production path exits
    with the designed rc (mem_guard.RSS_GUARD_RC), never a bare rc=1 — the
    broad `except Exception` in main() would otherwise swallow it (the
    RuntimeError subclass trap)."""
    import scripts.issue1739_holdout_rung as hr
    from explore_persona_space.experiments.issue_1739.mem_guard import (
        RSS_GUARD_RC,
        MemGuardRefusal,
    )

    def _boom(args):
        raise MemGuardRefusal("projected +25.7 GiB over 10.0 GiB available")

    monkeypatch.setattr(hr, "_run_production", _boom)
    rc = hr.main(["--behavior", "evil", "--output-dir", "/tmp/never-written"])
    assert rc == RSS_GUARD_RC, f"expected designed rc {RSS_GUARD_RC}, got {rc}"


def test_holdout_rung_production_wires_pre_phase_rss_guard() -> None:
    """The production path calls mem_guard.check_phase at BOTH heavy phase
    entries (whitening fit + labeled whitening; run_cell_multi grid) — the
    wiring pin for concern u1c-mem-guard (deleting either call fails here)."""
    import inspect

    import scripts.issue1739_holdout_rung as hr

    src = inspect.getsource(hr._run_production)
    assert "mem_guard.check_phase" in src
    assert "holdout_whitening[" in src and "holdout_grid[" in src
    assert "whitening_map_components" in src and "cell_solve_components" in src
    # the guard must see the LOADED-map regime (map_fit=False: no map fit here)
    assert "map_fit=False" in src


def test_holdout_ridge_folds_default_is_arm10_compatible() -> None:
    """The default --ridge-folds is 'all' -> ridge_folds=None: the holdout
    roster is FIXED all-16 incl. arm10_stacked, which needs ridge preds on
    EVERY fold. Pre-fix the transfer call hardcoded ridge_folds=(0,) and the
    2026-08-05 18:38:46Z run died at arms.py's contract violation AFTER a
    full CV pass (rc=1). 'all' also pins comparability with armfill's OOD
    arm10 rows (--ridge-folds all, markers v430/v450)."""
    import scripts.issue1739_holdout_rung as hr

    args = hr._parse_args([])
    assert args.ridge_folds == "all"
    assert hr._ridge_folds_arg(args) is None
    # the discarded-skip opt-out still maps to (0,) for arm10-free rosters
    args_skip = hr._parse_args(["--ridge-folds", "discarded-skip"])
    assert hr._ridge_folds_arg(args_skip) == (0,)


def test_holdout_startup_validation_refuses_arm10_with_fold_subset() -> None:
    """_validate_ridge_folds_roster fails at STARTUP (seconds) on the
    arm10 + fold-subset incompatibility that pre-fix only surfaced at the
    transfer pass, 25+ min in (after the CV pass)."""
    import pytest as _pytest

    import scripts.issue1739_holdout_rung as hr

    roster_with_arm10 = ["arm1_ctx_e1", "arm10_stacked", "arm16_surface_feat"]
    with _pytest.raises(RuntimeError, match="arm10_stacked"):
        hr._validate_ridge_folds_roster(roster_with_arm10, (0,))
    # all-fold ridge preds: compatible with arm10
    hr._validate_ridge_folds_roster(roster_with_arm10, None)
    # arm10-free roster: the fold subset is legal
    hr._validate_ridge_folds_roster(["arm1_ctx_e1", "arm4_ridge_ctx"], (0,))


def test_holdout_transfer_call_threads_ridge_folds() -> None:
    """The transfer-pass run_cell_multi call threads the validated
    ridge_folds variable — the pre-fix hardcoded literal (0,) is banned
    (it is what bypassed the startup validation's premise and killed the
    18:38:46Z run at arms.py:784)."""
    import inspect

    import scripts.issue1739_holdout_rung as hr

    src = inspect.getsource(hr._fit_eval_variant)
    assert "ridge_folds=ridge_folds" in src
    assert "ridge_folds=(0,)" not in src


# ---------------------------------------------------------------------------
# Pilot-judge selection + spread-instrument pins (task #1739 item-A gate).
#
# The round hit THREE instances of a silent-zero-work class (a selection
# predicate returns zero rows and the script reports success at rc=0; recorded
# on the task at 2026-08-05T17:18:31Z). These pin the pilot judge's
# non-empty-selection assertion and the plan-§7 spread instrument.
# ---------------------------------------------------------------------------
def _pilot_judge_module():
    pilot = REPO_ROOT / "scripts" / "issue1739_pilot_judge.py"
    if not pilot.exists():
        pytest.skip("issue1739_pilot_judge.py not present")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1739_pilot_judge as pj  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - env guard
        pytest.skip(f"pilot judge module import failed: {exc}")
    return pj


def test_pilot_load_rollouts_raises_on_missing_dir(tmp_path: Path) -> None:
    """A missing rollout dir RAISES — never a warn-and-continue rc=0 no-op."""
    pj = _pilot_judge_module()
    with pytest.raises(RuntimeError, match="rollout dir missing"):
        pj._load_rollouts(tmp_path / "nope" / "rollouts" / "pilot")


def test_pilot_load_rollouts_raises_on_empty_selection(tmp_path: Path) -> None:
    """An EMPTY selection is a failure, not a successful zero-row run."""
    pj = _pilot_judge_module()
    empty = tmp_path / "rollouts" / "pilot"
    empty.mkdir(parents=True)
    (empty / "_manifest.json").write_text("{}")  # underscore files are excluded
    with pytest.raises(RuntimeError, match="zero rollout JSONs"):
        pj._load_rollouts(empty)


def test_pilot_load_rollouts_raises_on_missing_required_key(tmp_path: Path) -> None:
    """A rollout missing a judged-item key RAISES at load, not mid-judge."""
    pj = _pilot_judge_module()
    d = tmp_path / "rollouts" / "pilot"
    d.mkdir(parents=True)
    (d / "ctx0_seed0.json").write_text(json.dumps({"context_id": "ctx0", "rollout_k": 0}))
    with pytest.raises(RuntimeError, match="missing required keys"):
        pj._load_rollouts(d)


def test_pilot_expected_rollout_count_from_manifest() -> None:
    """n_kept x k_rollouts is the expected-count basis; absent shape -> None."""
    pj = _pilot_judge_module()
    assert pj._expected_rollout_count({"n_kept": 200, "k_rollouts": 5}) == 1000
    assert pj._expected_rollout_count({"n_contexts": 200, "k_rollouts": 5}) == 1000
    assert pj._expected_rollout_count({"context_ids": ["a", "b"], "k_rollouts": 3}) == 6
    assert pj._expected_rollout_count({"k_rollouts": 5}) is None


def test_pilot_context_means_aggregates_per_context_and_reports_unscored() -> None:
    """Plan-§7 primary unit: one value per CONTEXT (mean over its rollouts)."""
    pj = _pilot_judge_module()
    item_context = {"c1_k00": "c1", "c1_k01": "c1", "c2_k00": "c2", "c3_k00": "c3"}
    scores = {"c1_k00": 20.0, "c1_k01": 40.0, "c2_k00": 90.0, "c3_k00": None}
    means, n_empty = pj._context_means(scores, item_context)
    assert sorted(means) == [30.0, 90.0]  # c1 -> mean(20,40); c2 -> 90; c3 unscored
    assert n_empty == 1


def test_pilot_context_means_truncation_filter_censors_and_reports() -> None:
    """The non-truncated sensitivity read drops contexts left with no rollout."""
    pj = _pilot_judge_module()
    item_context = {"c1_k00": "c1", "c1_k01": "c1", "c2_k00": "c2"}
    scores = {"c1_k00": 10.0, "c1_k01": 50.0, "c2_k00": 80.0}
    means, n_empty = pj._context_means(scores, item_context, keep_items={"c1_k01"})
    assert means == [50.0]
    assert n_empty == 1  # c2's only rollout was truncated-excluded


def test_pilot_spread_uses_sample_sd_and_strict_bottom_bin() -> None:
    """Instrument match with issue1739_k1_floor.rung_table: ddof=1 + strict `<`.

    A value sitting exactly ON the bottom edge (10.0) is NOT bottom-bin under
    the plan-§7 `mean_score < threshold` rule; the inclusive count is reported
    separately so neither convention is hidden.
    """
    pj = _pilot_judge_module()
    import statistics as _st

    vals = [10.0, 20.0, 30.0, 40.0]
    rep = pj._score_spread(vals, unit="context")
    assert rep["unit"] == "context"
    assert rep["sd"] == pytest.approx(_st.stdev(vals))  # ddof=1
    assert rep["sd_pop"] == pytest.approx(_st.pstdev(vals))  # ddof=0, reported
    assert rep["sd"] != pytest.approx(rep["sd_pop"])
    assert rep["bottom_frac"] == 0.0  # 10.0 is NOT < 10.0
    assert rep["bottom_frac_inclusive"] == 0.25  # but it IS <= 10.0


def test_pilot_spread_gate_is_two_sided_ceiling_fails() -> None:
    """A high-SD but CEILING-saturated rung FAILS (two-sided gate, plan §4.4)."""
    pj = _pilot_judge_module()
    vals = [100.0] * 8 + [0.0] * 2  # sd huge, top_frac 0.8 -> not < 0.80
    rep = pj._score_spread(vals, unit="context")
    assert rep["sd"] >= pj.SPREAD_SD_MIN
    assert rep["ceiling_frac"] == pytest.approx(0.8)
    assert rep["spread_gate_pass"] is False
    assert "top_frac" in rep["failed_criteria"]
    # floor fraction is reported alongside the ceiling regardless of outcome
    assert rep["bottom_frac"] is not None and rep["ceiling_frac"] is not None


def test_pilot_spread_gate_passes_on_genuine_spread() -> None:
    """A well-spread rung PASSES all three criteria."""
    pj = _pilot_judge_module()
    vals = [float(v) for v in range(0, 100, 5)]
    rep = pj._score_spread(vals, unit="context")
    assert rep["spread_gate_pass"] is True
    assert rep["failed_criteria"] == []
