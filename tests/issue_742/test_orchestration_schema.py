# ruff: noqa: RUF002, RUF003
"""Issue #742 round-2 regression tests — orchestration ↔ real #658 schema.

The round-1 review (epm:code-review v1 + codex v1) found 6 orchestration defects
that ALL slipped through 34 green library-binding tests: the tests bound the
estimators, not the scripts' integration with the REAL #658 on-disk schema. These
tests close that gap by exercising each orchestration load/judge/refit path against
a fixture that MIRRORS the real #658 shape (``cells[i].completions[j]["text"]``, the
real per-probe dict, the real ``analyzer_body_data.json`` ``/<genre>/a33/<beh>/layer``
keys), plus two counting/raising proofs:

  * judge-rerun-completion-key-crash + judge-rerun-wrong-judge-construct +
    judge-rerun-j-sampling: a 100-completion cell is sampled to EXACTLY J=20 and judged
    with the PER-BEHAVIOR construct (counting mock proves both).
  * join-integrity (v8 [REPLAN]): the DETERMINISTIC per-genre ``probe_pool_hash`` assert
    in ``load_inputs`` RAISES on a Betley↔UltraChat swap fixture (that hash, not a
    numeric ρ reproduction, is THE join gate), while the LOCO-CV ridge re-fit is a
    RECORDED DIAGNOSTIC — ``ridge_join_integrity`` writes refit_rho + persisted_rho +
    delta into the bracket entry and NEVER raises, even on a large (≈0.5) delta. This
    REPLACES the v7 unsatisfiable ``|refit − projection| ≤ tol → raise`` gate.
  * stage1-routing-layer: the A3.3 per-behavior layer is read from
    analyzer_body_data.json (Betley sycophancy 27, UltraChat refusal 6), NEVER the
    layer-21 locked_recipe fallback.

Determinism: 742X-family seeds (plan v7 §10).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from .conftest import impl_has  # noqa: E402

dc = importlib.import_module("explore_persona_space.analysis.issue_742_decoding_ceiling")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_658"


# --------------------------------------------------------------------------- #
# Fixtures mirroring the REAL #658 on-disk schema                              #
# --------------------------------------------------------------------------- #
def _real_e0_gen_cell(*, context_id: str, behavior: str, n_probes: int, n_rollouts: int) -> dict:
    """A gen-shaped dict matching the REAL #658 e0_gen schema.

    ``{context_id, column_id, dv, n_samples, cells: [{probe, completions: [{text,
    logp_norm}, ...]}, ...]}`` — completions carry ``text`` (NOT ``completion``; the
    judge-rerun-completion-key-crash BLOCKER was reading the wrong key).
    """
    cells = [
        {
            "probe": f"probe_{p}",
            "completions": [
                {"text": f"response {p}.{r} for {behavior}", "logp_norm": -1.0 - 0.01 * r}
                for r in range(n_rollouts)
            ],
        }
        for p in range(n_probes)
    ]
    return {
        "context_id": context_id,
        "column_id": behavior,
        "dv": "judged_rate",
        "n_samples": n_rollouts,
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
# 1. completion-key + per-behavior construct + J=20 sampling (BLOCKERs 1/2/3)  #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("sample_completions_for_judge"), reason="round-2 symbol")
def test_sample_completions_reads_text_key_and_caps_at_j():
    """100-completion cell → EXACTLY J=20 sampled, reading the real ``text`` key."""
    gen = _real_e0_gen_cell(
        context_id="f1_house_x", behavior="sycophancy", n_probes=10, n_rollouts=10
    )
    # 100 completions total; sample to J=20 deterministically
    sampled = dc.sample_completions_for_judge(gen, j_completions=20, seed=7428)
    total = sum(len(c["completions"]) for c in sampled["cells"])
    assert total == 20, f"expected exactly J=20 sampled completions, got {total}"
    assert sampled["n_sampled_for_judge"] == 20
    # every sampled completion carries the real 'text' key (NOT 'completion')
    for cell in sampled["cells"]:
        for comp in cell["completions"]:
            assert "text" in comp and "completion" not in comp
    # determinism: same seed -> same sample
    again = dc.sample_completions_for_judge(gen, j_completions=20, seed=7428)
    texts1 = [c["completions"][0]["text"] for c in sampled["cells"]]
    texts2 = [c["completions"][0]["text"] for c in again["cells"]]
    assert texts1 == texts2


@pytest.mark.skipif(not impl_has("per_behavior_judge_rate"), reason="round-2 symbol")
def test_per_behavior_judge_uses_correct_rubric_and_text_key():
    """The PER-BEHAVIOR construct is dispatched (NOT the default mean_aligned judge).

    Counting mock proves: (a) the J-sampled completions are judged via the behavior's
    OWN rubric (judge_fn receives the behavior col_id), (b) the completion ``text`` is
    what flows through, (c) the reconstructed rate is the judge-positive fraction —
    the SAME judged_rate construct #658 used, NOT a 0-100 alignment mean.
    """
    gen = _real_e0_gen_cell(context_id="f1_house_x", behavior="broad_em", n_probes=5, n_rollouts=4)
    sampled = dc.sample_completions_for_judge(gen, j_completions=20, seed=7428)
    calls: list[tuple[str, int]] = []

    def _counting_judge(col_id: str, g: dict, model: str) -> dict:
        # prove the per-behavior col_id is threaded + count the judged completions
        n = sum(len(c["completions"]) for c in g["cells"])
        # prove the real text key is present (would KeyError on the old c["completion"])
        for c in g["cells"]:
            for comp in c["completions"]:
                _ = comp["text"]
        calls.append((col_id, n))
        # half judged-positive -> rate 0.5 (a judged_rate, not a 0-100 mean)
        return {"column_id": col_id, "rate": 0.5, "n_judged": n, "n_positive": n // 2}

    res = dc.per_behavior_judge_rate(
        sampled,
        behavior="broad_em",
        judge_model="claude-sonnet-4-5-20250929",
        judge_fn=_counting_judge,
    )
    assert calls and calls[0][0] == "broad_em", "per-behavior rubric col_id must be threaded"
    assert calls[0][1] == 20, f"judge must see EXACTLY J=20 completions, saw {calls[0][1]}"
    assert res["rate"] == 0.5 and "n_positive" in res, "must return the judged_rate construct"


@pytest.mark.skipif(not impl_has("per_behavior_judge_rate"), reason="round-2 symbol")
def test_per_behavior_judge_rejects_non_readout_behavior():
    """A non-read-out behavior raises (no silent default-judge substitution)."""
    gen = _real_e0_gen_cell(context_id="c", behavior="deception", n_probes=2, n_rollouts=2)
    with pytest.raises(KeyError):
        dc.per_behavior_judge_rate(
            gen, behavior="deception", judge_model="m", judge_fn=lambda *a: {}
        )


# --------------------------------------------------------------------------- #
# 2. A3.3 per-behavior layer from analyzer_body_data.json (BLOCKER 6b)         #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (impl_has("load_a33_layer") and (EVAL_DIR / "analyzer_body_data.json").exists()),
    reason="round-2 symbol / #658 artifact",
)
def test_a33_layer_read_from_real_analyzer_body_data():
    """The Stage-1 layer comes from /<genre>/a33/<beh>/layer, NOT a layer-21 fallback."""
    # the plan's named expectations (verified against the real artifact this session)
    assert dc.load_a33_layer("sycophancy", "betley", eval_dir=EVAL_DIR) == 27
    assert dc.load_a33_layer("refusal", "ultrachat", eval_dir=EVAL_DIR) == 6
    # the per-behavior layers are genuinely heterogeneous (NOT a single default)
    betley_layers = {
        b: dc.load_a33_layer(b, "betley", eval_dir=EVAL_DIR) for b in dc.READOUT_BEHAVIORS
    }
    assert len(set(betley_layers.values())) > 1, f"layers must vary per behavior: {betley_layers}"
    # a missing key raises (no silent fallback)
    with pytest.raises(KeyError):
        dc.load_a33_layer("not_a_behavior", "betley", eval_dir=EVAL_DIR)


# --------------------------------------------------------------------------- #
# 3. v8 [REPLAN] join-integrity = probe_pool_hash assert; ridge = recorded diag #
#    (test 6 — replaces the v7 unsatisfiable |refit − projection| ≤ tol raise)  #
# --------------------------------------------------------------------------- #
def _write_v0_fixture(repo_root: Path, genre: str, *, probe_pool_hash: str) -> None:
    """Write a minimal #658-shaped v0_summaries.pt under the genre's expected path.

    ``summaries[recipe]`` is the real ``dict[context_id -> Tensor(28, 3584)]`` shape
    (MF1, §12 row 1) so a faithful load gets PAST the hash check; the ``probe_pool_hash``
    is what the join-integrity assert keys on.
    """
    import torch

    rel = dc.GENRE_V0_PATHS[genre]
    out = repo_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    context_ids = [f"ctx_{i}" for i in range(50)]
    summaries = {
        recipe: {c: torch.zeros(28, 3584) for c in context_ids}
        for recipe in ("mean", "last", "maxp")
    }
    torch.save(
        {
            "summaries": summaries,
            "context_ids": context_ids,
            "capture_layers": list(range(28)),
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "probe_pool_hash": probe_pool_hash,
        },
        out,
    )


@pytest.mark.skipif(not impl_has("load_inputs"), reason="round-2 symbol")
def test_probe_pool_hash_raises_on_swap_and_ridge_delta_reported(tmp_path):
    """v8 [REPLAN] join-integrity contract (plan §14 test 6).

    (a) ``load_inputs`` RAISES via the DETERMINISTIC per-genre ``probe_pool_hash`` assert
        when the loaded Betley tensor carries the UltraChat hash (a Betley↔UltraChat
        swap) — that hash, not a numeric ρ reproduction, is THE join gate. A faithful
        load (matching hash) gets PAST the hash check.
    (b) ``ridge_join_integrity`` is a RECORDED DIAGNOSTIC: ``compute_bracket`` writes
        ``refit_rho``/``persisted_rho``/``delta`` into the bracket entry and DOES NOT
        raise even when the delta is large (~0.5, the sycophancy regime) — re-introducing
        a ``delta > tol -> raise`` would make this fail, catching a revert to the v7 gate.
    """
    betley_hash = dc.GENRE_EXPECTED_PROBE_POOL_HASH["betley"]
    ultrachat_hash = dc.GENRE_EXPECTED_PROBE_POOL_HASH["ultrachat"]

    # (a.1) SWAP: Betley path holds the UltraChat hash -> ValueError naming the hash.
    _write_v0_fixture(tmp_path, "betley", probe_pool_hash=ultrachat_hash)
    with pytest.raises(ValueError, match="probe_pool_hash"):
        dc.load_inputs("betley", repo_root=tmp_path)

    # (a.2) FAITHFUL: Betley path holds the Betley hash -> gets PAST the hash assert.
    #       (a later miss on the E0/a33 artifacts is fine and is NOT a hash error — we
    #       assert only that the failure, if any, is no longer the probe_pool_hash one.)
    faithful_root = tmp_path / "faithful"
    _write_v0_fixture(faithful_root, "betley", probe_pool_hash=betley_hash)
    try:
        dc.load_inputs("betley", repo_root=faithful_root)
    except ValueError as exc:  # pragma: no cover - defensive
        assert "probe_pool_hash" not in str(exc), (
            "a hash-matched faithful load must get PAST the probe_pool_hash assert"
        )
    except (FileNotFoundError, KeyError):
        pass  # expected: no E0/a33 artifacts staged in the tmp fixture

    # (b) Large-delta ridge re-fit is REPORTED, never raised on. Drive the orchestration
    # path (compute_bracket) with a synthetic cell whose ridge re-fit lands ~0.5 from the
    # persisted projection (the sycophancy regime). A linearly-decodable target gives a
    # real held-out ridge ρ; persisted_rho is set far below it so |delta| ~ 0.5.
    import issue742_reliability as rel

    rng = np.random.default_rng(74240)
    n, d = 50, 12
    v0 = rng.normal(0, 1, size=(n, d))
    w = rng.normal(0, 1, size=d)
    rates = 1.0 / (1.0 + np.exp(-(v0 @ w)))  # in (0,1), linearly decodable
    refit_rho = dc.loco_ridge_refit_rho(v0, rates)
    assert refit_rho > 0.3, f"decodable target must refit to a real held-out rho, got {refit_rho}"
    persisted_far = max(0.0, refit_rho - 0.5)  # force a large delta (~0.5)

    context_ids = [f"ctx_{i}" for i in range(n)]
    # #658-shaped per-context cell: rate + n_judged + per_probe list of {probe, e0, n_judged}
    e0 = {
        c: {
            "sycophancy": {
                "rate": float(rates[i]),
                "n_judged": 200,
                "per_probe": [
                    {"probe": f"p{j}", "e0": float(rates[i]), "n_judged": 1} for j in range(4)
                ],
            }
        }
        for i, c in enumerate(context_ids)
    }

    entry = rel.compute_bracket(  # MUST NOT raise on a large delta
        "sycophancy",
        "betley",
        e0,
        context_ids,
        rho_lin=persisted_far,
        rng=rng,
        n_split_seeds=5,
        n_boot=10,
        v0_layer=v0,
        layer=0,
        join_tol=0.05,
    )
    ji = entry["ridge_join_integrity"]
    assert ji is not None, "the ridge_join_integrity diagnostic must be recorded"
    for key in ("refit_rho", "persisted_rho", "delta"):
        assert key in ji, f"diagnostic must carry {key}"
    assert ji["delta"] > 0.05, f"this fixture is a large-delta cell, got delta={ji['delta']}"
    assert ji["join_ok"] is False, "join_ok is a reported flag (large delta -> False), not a gate"

    # The orchestration `run` path must not raise on a large-delta entry either: there is
    # no `if join_failures: raise` block left to re-introduce the v7 gate.
    src = (PROJECT_ROOT / "scripts" / "issue742_reliability.py").read_text()
    assert "raise RuntimeError" not in src or "join-integrity" not in src, (
        "the v7 join-integrity RuntimeError must be gone (ridge is a diagnostic, not a gate)"
    )


# --------------------------------------------------------------------------- #
# 4. CV-matched reliability CI is fold-matched, not pooled (BLOCKER 5a)        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("cv_matched_reliability_ci"), reason="round-2 symbol")
def test_cv_matched_reliability_excludes_one_context_per_fold():
    """The CV-matched CI is computed over n LOCO folds (one held-out ctx each), so a
    single outlier context moves exactly one fold estimate — a pooled bootstrap would
    mix it into every resample. We assert the fold spread responds to a planted
    outlier (the fold-matched signature)."""
    rng = np.random.default_rng(74250)
    n = 50
    rates = rng.uniform(0.2, 0.8, size=n)
    m = np.full(n, 200.0)
    mean0, lo0, hi0 = dc.cv_matched_reliability_ci(rates, m)
    # plant one extreme-variance context; the fold that EXCLUDES it differs from folds
    # that include it -> the across-fold spread widens (fold-matched, not pooled)
    rates2 = rates.copy()
    rates2[0] = 0.999
    mean1, lo1, hi1 = dc.cv_matched_reliability_ci(rates2, m)
    assert 0.0 <= lo0 <= hi0 <= 1.0 and 0.0 <= lo1 <= hi1 <= 1.0
    # the planted outlier must move the estimate (proves it is data-driven per fold)
    assert abs(mean1 - mean0) > 1e-6


# --------------------------------------------------------------------------- #
# 5. dcor_at_subsample is well-posed at small n' (BLOCKER 7 dCor(n') curve)    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("dcor_at_subsample"), reason="round-2 symbol")
def test_dcor_at_subsample_is_bounded_and_clamps_d_eff():
    """dCor(n') returns a value in [0,1] even when d_eff > n'-1 (it clamps)."""
    rng = np.random.default_rng(74260)
    v0 = rng.normal(0, 1, size=(50, 30))
    e0 = rng.uniform(0, 1, size=50)
    val = dc.dcor_at_subsample(v0, e0, n_prime=10, d_eff=20, rng=rng)
    assert 0.0 <= val <= 1.0


# --------------------------------------------------------------------------- #
# 6. §7 estimator-disagreement guard ALIVE for heterogeneous-probe behaviors   #
#    (CONCERN one-rollout-splithalf-still-dropped) — drives compute_bracket    #
#    end-to-end on a RAGGED-probe fixture (the PRODUCTION path, NOT             #
#    load_reliability_estimates which no script calls).                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("reliability_split_half_over_probes"), reason="round-2 symbol")
def test_compute_bracket_runs_split_half_on_heterogeneous_probe_counts():
    """The round-2 §7 disagreement guard was DEAD for harmful_compliance/refusal:
    ``_build_probe_rate_matrix`` returned ``None`` on heterogeneous per-context probe
    counts, so ``compute_bracket`` silently dropped the split-half estimator and only
    the binomial read stood — for exactly the two behaviors the guard protects.

    This binds the PRODUCTION path (``compute_bracket``, the path the script calls),
    NOT ``load_reliability_estimates`` (a library helper no script calls — the prior
    round's test bound the wrong path). It feeds a #658-shaped ``e0`` dict with
    heterogeneous probe counts ({114, 115} harmful_compliance; {212..215} refusal) and
    asserts BOTH the split-half AND the binomial estimators are populated, with the
    realized truncation depth recorded.
    """
    import issue742_reliability as rel

    rng = np.random.default_rng(74270)

    def _build_ragged_e0(behavior: str, probe_counts_cycle: list[int]) -> tuple[dict, list[str]]:
        # 50 contexts; each context's per_probe list length cycles through the
        # heterogeneous probe counts (so the matrix is ragged, exactly the real shape).
        n = 50
        context_ids = [f"ctx_{i}" for i in range(n)]
        # a genuine per-context signal so the estimators have something to recover
        theta = np.clip(0.5 + rng.normal(0, 0.12, n), 0.05, 0.95)
        e0: dict = {}
        for i, c in enumerate(context_ids):
            n_probes = probe_counts_cycle[i % len(probe_counts_cycle)]
            probe_labels = rng.binomial(1, theta[i], size=n_probes).astype(float)
            e0[c] = {
                behavior: {
                    "rate": float(probe_labels.mean()),
                    "n_judged": int(n_probes),
                    "per_probe": [
                        {"probe": f"p{j}", "e0": float(probe_labels[j]), "n_judged": 1}
                        for j in range(n_probes)
                    ],
                }
            }
        return e0, context_ids

    for behavior, counts in (
        ("harmful_compliance", [114, 115]),
        ("refusal", [212, 213, 214, 215]),
    ):
        e0, context_ids = _build_ragged_e0(behavior, counts)
        entry = rel.compute_bracket(
            behavior,
            "betley",
            e0,
            context_ids,
            rho_lin=0.2,
            rng=np.random.default_rng(74271),
            n_split_seeds=20,
            n_boot=10,
        )
        # BOTH estimators must be populated (the split-half was None before the fix)
        assert entry["r_yy_split_half"] is not None, (
            f"{behavior}: split-half MUST run on heterogeneous probe counts "
            f"(was silently dropped -> §7 guard dead); estimator_kind={entry['estimator_kind']!r}"
        )
        assert entry["r_yy_binomial"] is not None
        assert entry["estimator_kind"] == "split_half_over_probes"
        # the truncation depth recorded == min(probe_counts) for the analyzer to see
        assert entry["split_half_m_actual_probes"] == min(counts), (
            f"{behavior}: m_actual must be min(probe_counts)={min(counts)}, "
            f"got {entry['split_half_m_actual_probes']}"
        )
        # the disagreement guard can now FIRE (it has a split_half to compare); the
        # boolean is data-dependent, but the field must be a real bool, not the
        # always-False dead value the dropped split-half produced.
        assert isinstance(entry["estimators_disagree"], bool)


# --------------------------------------------------------------------------- #
# 7. J-sampling per-cell seed is CROSS-PROCESS STABLE                          #
#    (BLOCKER judge-rerun-nondeterministic-sampling) — Python's salted hash()  #
#    produced different seeds in different interpreter processes; the sha256    #
#    digest seed is identical everywhere.                                      #
# --------------------------------------------------------------------------- #
def test_stable_cell_seed_is_identical_across_interpreter_processes():
    """``_stable_cell_seed`` must return the SAME value in two fresh interpreters with
    DIFFERENT ``PYTHONHASHSEED`` — the salted builtin ``hash()`` did not (Codex showed
    102742103 vs 2626216375 for the same tuple), so a fixed ``--seed`` would NOT
    reproduce the same J=20 sample. Spawns two subprocesses with distinct hash seeds
    and asserts equality (and that the builtin hash genuinely DIFFERS, proving the
    fixture exercises the salted-hash regime).
    """
    import subprocess

    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); "
        "from issue742_judge_rerun import _stable_cell_seed; "
        "print(_stable_cell_seed(7428, 'betley', 'refusal', 'f1_house_surgeon')); "
        "print(abs(hash((7428, 'betley', 'refusal', 'f1_house_surgeon'))) % (2**32))"
    )

    def _run(hashseed: str) -> tuple[int, int]:
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        out = subprocess.run(
            ["uv", "run", "python", "-c", snippet],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        return int(lines[0]), int(lines[1])

    stable_a, builtin_a = _run("0")
    stable_b, builtin_b = _run("12345")

    assert stable_a == stable_b, (
        f"_stable_cell_seed must be cross-process stable: {stable_a} != {stable_b}"
    )
    assert builtin_a != builtin_b, (
        "fixture sanity: the salted builtin hash() must DIFFER across the two "
        f"PYTHONHASHSEEDs (got {builtin_a} == {builtin_b}) — else this test would not "
        "exercise the nondeterminism the BLOCKER is about"
    )


# --------------------------------------------------------------------------- #
# 8. The LIVE (non-dry) judge-rerun path is exercised end-to-end, no API spend #
#    (BLOCKER judge-rerun-smoke-dry-run-only) — drives run() with a counting    #
#    judge over a seeded synthetic snapshot; asserts a real judge_variance.    #
# --------------------------------------------------------------------------- #
def test_nondry_counting_judge_writes_real_judge_variance(tmp_path):
    """The dry-run smoke NEVER reaches ``_judge_reruns_for_cell`` /
    ``_decompose_variance`` / the ``judge_variance`` write. This drives the LIVE
    (``dry_run=False``) ``run`` path with the deterministic counting judge over a
    pre-seeded synthetic snapshot (no HF, no #658 tensors, no API spend) and asserts
    the variance decomposition actually computed and was written: ``judge_variance``
    is non-empty and carries a NON-ZERO ``var_judge`` (the across-rerun term — a dead
    path would write nothing or a degenerate zero).
    """
    import issue742_judge_rerun as jr

    dest = tmp_path / "snap"
    jr.seed_synthetic_snapshot(dest, genre="betley", behavior="refusal")
    result = jr.run(
        genres=["betley"],
        behaviors=["refusal"],
        r_rerun=2,
        j_completions=20,
        dry_run=False,
        seed=7428,
        judge_fn=jr.make_counting_judge(),
        dest_override=dest,
        skip_snapshot=True,
    )
    jv = result["judge_variance"]
    assert jv, "judge_variance must be non-empty (the live path must have run)"
    decomp = jv["betley"]["refusal"]
    for key in ("var_total", "var_judge", "var_generation", "var_signal", "sqrt_r_yy_honest"):
        assert key in decomp, f"variance decomposition missing {key}"
    assert decomp["var_judge"] > 0.0, (
        f"var_judge must be NON-ZERO to demonstrate the across-rerun term computing, "
        f"got {decomp['var_judge']}"
    )
    assert "non-dry" in result["note"], "the note must record the non-dry counting-judge run"


# --------------------------------------------------------------------------- #
# 9. [REPLAN-FOLLOWUP-2] Stage-1 verdict CONSUMES the control-task null         #
#    (plan v9 §4 Stage-1 step 4 / §14 test 7; concern                           #
#    stage1-verdict-ignores-control-task-null). A both-pass control (d≫n probe  #
#    memorization) must NOT read nonlinear-yes — it reads non-selective.        #
# --------------------------------------------------------------------------- #
def _dcor_result(*, dcor: float, p_value: float):
    """Build a minimal DcorPermutationResult stand-in for the monkeypatched test."""
    return dc.DcorPermutationResult(
        dcor=dcor,
        null=np.zeros(3, dtype=float),
        p_value=p_value,
        d_eff=10,
        n_perm=3,
    )


# (a) the verdict-classifier rule directly (plan §14 test 7a) ---------------- #
@pytest.mark.skipif(not impl_has("classify_stage1_verdict"), reason="round-5 symbol")
def test_classify_stage1_verdict_selectivity_paths():
    """The 5-entry enum + the three selective/non-selective limbs (plan v9 §11 row 11).

    nonlinear-yes requires dcor_pass AND linear_pass AND selective, where selective =
    control fails its own null (path i) OR observed beats control by ≥ Δ_sel (path ii).
    A both-pass control with a sub-Δ_sel margin reads non-selective, NEVER nonlinear-yes.
    The legacy ceiling-limited / linear-erasure returns are unchanged.
    """
    # path (i): control FAILS its own null (p >= α) -> selective -> nonlinear-yes
    ctrl_fails = _dcor_result(dcor=0.30, p_value=0.50)
    v_i = dc.classify_stage1_verdict(
        dcor_pass=True, linear_pass=True, control_res=ctrl_fails, observed_dcor=0.50
    )
    assert v_i == "nonlinear-yes"
    assert v_i.selectivity_branch == "control-fails-null"

    # path (ii): control ALSO passes (p < α) but observed beats it by ≥ Δ_sel=0.10
    ctrl_passes_lowdcor = _dcor_result(dcor=0.30, p_value=0.001)
    v_ii = dc.classify_stage1_verdict(
        dcor_pass=True, linear_pass=True, control_res=ctrl_passes_lowdcor, observed_dcor=0.50
    )
    assert v_ii == "nonlinear-yes"
    assert v_ii.selectivity_branch == "effect-size-margin"

    # NOT selective: control passes AND margin < Δ_sel -> non-selective (d≫n memorization)
    ctrl_passes_close = _dcor_result(dcor=0.45, p_value=0.001)
    v_ns = dc.classify_stage1_verdict(
        dcor_pass=True, linear_pass=True, control_res=ctrl_passes_close, observed_dcor=0.50
    )
    assert v_ns == "non-selective"
    assert v_ns != "nonlinear-yes"
    assert v_ns.selectivity_branch == "non-selective"

    # legacy branches unchanged
    assert (
        dc.classify_stage1_verdict(dcor_pass=True, linear_pass=False, control_res=ctrl_fails)
        == "linear-erasure-leakage-unresolved"
    )
    assert dc.classify_stage1_verdict(dcor_pass=False, linear_pass=True) == "ceiling-limited"

    # back-compat: control_res=None passes selectivity vacuously (legacy 2-arg contract)
    assert dc.classify_stage1_verdict(dcor_pass=True, linear_pass=True) == "nonlinear-yes"

    # the 5-entry enum is exactly the plan v9 §6.5 set
    assert (
        frozenset(
            {
                "nonlinear-yes",
                "non-selective",
                "linear-erasure-leakage-unresolved",
                "ceiling-limited",
                "indistinguishable-from-null-given-variance",
            }
        )
        == dc.STAGE1_VERDICTS
    )


# Shared monkeypatch harness driving run_stage1_cell with controlled dCor reads. #
def _patch_run_stage1_cell(monkeypatch, *, true_res, control_res, linear_pass: bool = True):
    """Patch the dCor + held-out + PCA calls so run_stage1_cell's verdict logic is
    exercised deterministically: the FIRST dcor_permutation_test call (true task)
    returns ``true_res``, the SECOND (shuffled control) returns ``control_res``."""
    calls = {"n": 0}

    def _fake_dcor(v0, E0, *, d_eff, n_perm, rng, **kw):
        calls["n"] += 1
        return true_res if calls["n"] == 1 else control_res

    monkeypatch.setattr(dc, "dcor_permutation_test", _fake_dcor)
    # held-out diagnostic: a passing (no residual linear leakage) result
    monkeypatch.setattr(
        dc,
        "held_out_linear_leakage",
        lambda *a, **k: dc.HeldOutLeakageResult(
            rho=0.0, null_ci=(-0.2, 0.2), post_leace_linear_pass=linear_pass
        ),
    )

    n, d = 50, 12
    rng = np.random.default_rng(74280)
    v0_layer = rng.normal(0, 1, size=(n, d))
    E0 = rng.uniform(0, 1, size=n)
    from issue742_nonlinear_residual import run_stage1_cell

    return run_stage1_cell(
        v0_layer,
        E0,
        behavior="sycophancy",
        genre="betley",
        layer=27,
        d_eff=10,
        n_perm=3,
        rng=rng,
        run_power_check=False,
        power_trials=3,
        power_perm=10,
    )


# (b) end-to-end memorization regression (plan §14 test 7b) ------------------ #
def test_run_stage1_cell_both_pass_control_is_non_selective(monkeypatch):
    """Both the true call AND the shuffled control pass at comparable dCor
    (|true − control| < Δ_sel=0.10): a d≫n probe-memorization cell. The verdict
    MUST be non-selective, NOT nonlinear-yes — the cap-3 defect (the script dropped
    control_res) would have read nonlinear-yes here. Regression guard for that revert.
    """
    true_res = _dcor_result(dcor=0.50, p_value=0.001)
    control_res = _dcor_result(dcor=0.45, p_value=0.001)  # passes, gap 0.05 < 0.10
    entry = _patch_run_stage1_cell(monkeypatch, true_res=true_res, control_res=control_res)
    assert entry["stage1_verdict"] == "non-selective"
    assert entry["stage1_verdict"] != "nonlinear-yes"
    assert entry["selectivity_rule_passed"] is False
    assert entry["selectivity_branch"] == "non-selective"


def test_run_stage1_cell_control_fails_null_is_nonlinear_yes(monkeypatch):
    """Path (i): the shuffled control FAILS its own null (p=0.5) -> selective ->
    nonlinear-yes with selectivity_branch=control-fails-null."""
    true_res = _dcor_result(dcor=0.50, p_value=0.001)
    control_res = _dcor_result(dcor=0.50, p_value=0.50)  # fails its null
    entry = _patch_run_stage1_cell(monkeypatch, true_res=true_res, control_res=control_res)
    assert entry["stage1_verdict"] == "nonlinear-yes"
    assert entry["selectivity_branch"] == "control-fails-null"
    assert entry["selectivity_rule_passed"] is True


def test_run_stage1_cell_effect_size_margin_is_nonlinear_yes(monkeypatch):
    """Path (ii): the control passes (p<α) but the true task beats it by ≥ Δ_sel=0.10
    -> nonlinear-yes with selectivity_branch=effect-size-margin."""
    true_res = _dcor_result(dcor=0.50, p_value=0.001)
    control_res = _dcor_result(dcor=0.30, p_value=0.001)  # passes, gap 0.20 >= 0.10
    entry = _patch_run_stage1_cell(monkeypatch, true_res=true_res, control_res=control_res)
    assert entry["stage1_verdict"] == "nonlinear-yes"
    assert entry["selectivity_branch"] == "effect-size-margin"
    assert entry["selectivity_rule_passed"] is True


# (c) defense-in-depth invariant (plan §14 test 7c) -------------------------- #
def test_run_stage1_cell_nonlinear_yes_implies_selectivity_holds(monkeypatch):
    """Whenever run_stage1_cell emits nonlinear-yes, the recorded reads must satisfy
    the selectivity rule literally (the per-cell output assert, mirrored as a test)."""
    true_res = _dcor_result(dcor=0.50, p_value=0.001)
    control_res = _dcor_result(dcor=0.50, p_value=0.50)
    entry = _patch_run_stage1_cell(monkeypatch, true_res=true_res, control_res=control_res)
    if entry["stage1_verdict"] == "nonlinear-yes":
        selective = entry["control_task_p_value"] >= 0.05 or (
            entry["control_task_p_value"] < 0.05
            and (entry["dcor_observed"] - entry["control_task_dcor"]) >= 0.10
        )
        assert entry["selectivity_rule_passed"] is True
        assert entry["post_leace_linear_pass"] is True
        assert entry["dcor_pass"] is True
        assert selective, "nonlinear-yes recorded but selectivity rule does not hold on the reads"


# --------------------------------------------------------------------------- #
# 9. HEADLINE CI = cluster-bootstrap, NEVER the CV-matched LOCO-fold percentile #
#    (BLOCKER stage0-bootstrap-ci-not-used) — binds the PRODUCTION path.        #
#    Plan v9 §4 Stage-0 step 4 + §11 row 6 + parameters table + §14 test plan   #
#    bind cluster_bootstrap_ci(B=2000, seed 742) for the headline √(r_yy) /     #
#    bracket-width CI; the prior round threaded n_boot to the API boundary but  #
#    headlined on cv_matched_reliability_ci (percentile-across-folds), biasing   #
#    the gate toward false "headroom". This monkeypatches BOTH helpers as spies  #
#    and asserts the headline CI is the bootstrap output, never the CV-fold one. #
# --------------------------------------------------------------------------- #
def test_headline_ci_is_cluster_bootstrap_not_cv_matched_fold(monkeypatch):
    """``compute_bracket`` MUST headline ``sqrt_r_yy_ci`` / ``bracket_width_ci`` on
    ``dc.cluster_bootstrap_ci`` (the B=n_boot cluster bootstrap over contexts), NEVER
    on ``dc.cv_matched_reliability_ci`` (the percentile-across-LOCO-folds read, now a
    recorded diagnostic). Spies on both helpers with distinct sentinel intervals and
    asserts (a) the bootstrap spy was called with the threaded ``n_boot``; (b) the
    headline CI equals the BOOTSTRAP sentinel, not the CV-fold sentinel."""
    import issue742_reliability as rel

    # 50-context #658-shaped e0 with a genuine per-context signal.
    rng = np.random.default_rng(74290)
    n = 50
    context_ids = [f"ctx_{i}" for i in range(n)]
    theta = np.clip(0.5 + rng.normal(0, 0.12, n), 0.05, 0.95)
    e0: dict = {}
    for i, c in enumerate(context_ids):
        m = 400
        labels = rng.binomial(1, theta[i], size=m).astype(float)
        e0[c] = {
            "broad_em": {
                "rate": float(labels.mean()),
                "n_judged": int(m),
                "per_probe": [
                    {"probe": f"p{j}", "e0": float(labels[j]), "n_judged": 1} for j in range(8)
                ],
            }
        }

    BOOT_CI = (0.111, 0.999)  # sentinel the cluster-bootstrap spy returns
    CV_CI = (0.222, 0.888)  # sentinel the CV-matched-fold spy returns
    boot_calls: list[int] = []

    def _spy_bootstrap(stat_fn, data, *, n_boot, alpha=0.05, rng=None):
        boot_calls.append(int(n_boot))
        # exercise stat_fn so a broken stat function still surfaces (it must be callable
        # on a resample of `data`'s rows).
        _ = float(stat_fn(np.asarray(data)))
        return BOOT_CI

    cv_calls: list[int] = []

    def _spy_cv(rates, m_cell, *, alpha=0.05):
        cv_calls.append(1)
        return (0.5, CV_CI[0], CV_CI[1])  # (mean, lo, hi)

    monkeypatch.setattr(rel.dc, "cluster_bootstrap_ci", _spy_bootstrap)
    monkeypatch.setattr(rel.dc, "cv_matched_reliability_ci", _spy_cv)

    entry = rel.compute_bracket(
        "broad_em",
        "betley",
        e0,
        context_ids,
        rho_lin=0.20,
        rng=np.random.default_rng(742),
        n_split_seeds=20,
        n_boot=137,  # distinctive value the bootstrap spy must receive
    )

    # (a) cluster_bootstrap_ci was called for the headline CI, with the threaded n_boot.
    assert boot_calls == [137], (
        f"cluster_bootstrap_ci must be called once with n_boot=137; got {boot_calls}"
    )

    # (b) the headline CI is the BOOTSTRAP sentinel — NOT the CV-matched-fold sentinel.
    assert entry["sqrt_r_yy_ci"] == [BOOT_CI[0], BOOT_CI[1]], (
        f"headline sqrt_r_yy_ci must be the cluster-bootstrap CI {list(BOOT_CI)}, "
        f"got {entry['sqrt_r_yy_ci']}"
    )
    assert entry["ceiling_ci_kind"] == "cluster_bootstrap"
    assert entry["n_boot"] == 137
    # bracket_width_ci is derived from the headline (bootstrap) ceiling bounds − rho_lin.
    assert entry["bracket_width_ci"] == [BOOT_CI[0] - 0.20, BOOT_CI[1] - 0.20]

    # the CV-matched fold read is still computed, but ONLY as a recorded diagnostic.
    assert cv_calls == [1], "cv_matched_reliability_ci should still run for the diagnostic"
    assert entry["sqrt_r_yy_cv_matched_fold_ci"] == [CV_CI[0], CV_CI[1]], (
        "CV-matched fold CI must be recorded as a diagnostic, distinct from the headline"
    )
    # the headline must NOT be the CV-matched read.
    assert entry["sqrt_r_yy_ci"] != entry["sqrt_r_yy_cv_matched_fold_ci"]


# --------------------------------------------------------------------------- #
# 10. BLOCKER variance-decomposition-wrong-grain — Var_generation is the        #
#     WITHIN-context generation variance (averaged across contexts), NOT the    #
#     between-context variance of a single half-mean (which is SIGNAL, the      #
#     term we want to RECOVER, and routinely exceeded var_total -> var_signal=0 #
#     -> sqrt_r_yy_honest=0). Plan v9 §4 Stage-0 step 3.                         #
# --------------------------------------------------------------------------- #
def _const_rerun_rates(per_context_full: np.ndarray, n_rerun: int = 2) -> list[np.ndarray]:
    """R IDENTICAL judge-rerun rate vectors -> Var_judge == 0 (judge noise isolated out
    so the test reads Var_generation cleanly). Each entry is the per-context full rate."""
    return [per_context_full.copy() for _ in range(n_rerun)]


def test_decompose_variance_no_generation_noise_gives_near_zero_var_gen():
    """Both generation halves share the SAME per-context signal (no within-context
    generation noise) -> Var_generation ≈ 0, NOT Var_C(first_half).

    The prior round computed var_gen = np.var(first_half), the BETWEEN-context variance
    of the half-mean rates. With both halves equal to a per-context signal that VARIES
    across contexts, that wrong formula returns the (large) between-context signal
    variance; the correct (p1−p2)²/4 within-context estimator returns ~0.
    """
    import issue742_judge_rerun as jr

    rng = np.random.default_rng(74300)
    # per-context full rate varies strongly across contexts (this is SIGNAL).
    full = np.clip(0.5 + rng.normal(0, 0.2, 50), 0.05, 0.95)
    # both halves EQUAL the per-context signal -> zero within-context generation noise.
    p1 = full.copy()
    p2 = full.copy()
    decomp = jr._decompose_variance(_const_rerun_rates(full), p1, p2)
    var_total = decomp["var_total"]
    var_gen = decomp["var_generation"]
    # Var_generation must be ~0 (no within-context noise) — NOT the between-context
    # signal variance the wrong np.var(first_half) formula would return (~var_total).
    assert var_gen < 1e-9, f"no-generation-noise fixture must give var_gen~0, got {var_gen}"
    assert var_gen < var_total, (
        f"var_gen ({var_gen}) must be << var_total ({var_total}); the wrong between-context "
        "formula would make var_gen ~= var_total (it IS the signal variance)"
    )
    # signal recovered, honest ceiling non-zero (the collapse-to-0 bug is gone).
    assert decomp["var_signal"] > 0.0
    assert decomp["sqrt_r_yy_honest"] > 0.0
    # var_judge == 0 by construction (identical reruns) — generation term isolated.
    assert decomp["var_judge"] == pytest.approx(0.0, abs=1e-12)


def test_decompose_variance_recovers_known_within_context_generation_noise():
    """A fixture with KNOWN within-context generation noise: each half is an
    independent Binomial draw of the same context rate. The (p1−p2)²/4 estimator must
    recover the full-sample generation variance ≈ mean_c[p_c(1−p_c)/n_full], NOT 0 and
    NOT the between-context variance."""
    import issue742_judge_rerun as jr

    rng = np.random.default_rng(74301)
    n_contexts = 200  # many contexts so the across-context average of the estimator is tight
    n_full = 40  # completions per context; each half has n_full//2 = 20
    half_n = n_full // 2
    theta = np.clip(0.5 + rng.normal(0, 0.12, n_contexts), 0.1, 0.9)

    p1 = np.empty(n_contexts)
    p2 = np.empty(n_contexts)
    full = np.empty(n_contexts)
    for c in range(n_contexts):
        x = rng.binomial(1, theta[c], size=n_full)
        full[c] = x.mean()
        p1[c] = x[:half_n].mean()
        p2[c] = x[half_n:].mean()

    decomp = jr._decompose_variance(_const_rerun_rates(full), p1, p2)
    # the planted full-sample generation variance is the mean binomial variance of the
    # full mean: mean_c[ theta_c(1-theta_c) / n_full ].
    planted_var_gen = float(np.mean(theta * (1.0 - theta) / n_full))
    # (p1-p2)^2/4 is an unbiased estimator of that full-sample generation variance.
    assert decomp["var_generation"] == pytest.approx(planted_var_gen, abs=0.0015), (
        f"var_generation {decomp['var_generation']:.5f} must recover the planted "
        f"within-context full-sample generation variance {planted_var_gen:.5f}"
    )
    # sanity: it is positive (real noise) and far below the wrong between-context read
    # of the half-means (which would be ~Var_C(p1), much larger).
    assert decomp["var_generation"] > 0.0
    wrong_between_context = float(np.var(p1))
    assert decomp["var_generation"] < wrong_between_context, (
        "the within-context estimator must be far below the between-context var of the "
        f"half-means (correct={decomp['var_generation']:.5f} vs wrong={wrong_between_context:.5f})"
    )


# --------------------------------------------------------------------------- #
# 11. BLOCKER bracket-mixed-estimator-impossible-output — compute_bracket never  #
#     emits a negative bracket_width with a positive CI / `headroom`; the point  #
#     always lies inside its own CI even when the judge-folded diagnostic is     #
#     BELOW rho_lin (the exact smoke artifact: judge_honest_sqrt < rho_lin).     #
#     Plan v9 §4 Stage-0 step 4.                                                  #
# --------------------------------------------------------------------------- #
def test_compute_bracket_no_impossible_output_when_judge_folded_below_rho_lin():
    """Drive compute_bracket with judge_honest_sqrt FAR BELOW rho_lin (the regime that
    produced bracket=[0.444, 0.0] / bracket_width=-0.444 / headroom). The headline must
    be the bootstrap-consistent binomial estimator, so: (a) bracket[0] <= bracket[1];
    (b) bracket_width is inside bracket_width_ci; (c) no negative width with headroom."""
    import issue742_reliability as rel

    rng = np.random.default_rng(74310)
    n = 50
    context_ids = [f"ctx_{i}" for i in range(n)]
    theta = np.clip(0.5 + rng.normal(0, 0.15, n), 0.05, 0.95)
    e0: dict = {}
    for i, c in enumerate(context_ids):
        m = 400
        labels = rng.binomial(1, theta[i], size=m).astype(float)
        e0[c] = {
            "broad_em": {
                "rate": float(labels.mean()),
                "n_judged": int(m),
                "per_probe": [
                    {"probe": f"p{j}", "e0": float(labels[j]), "n_judged": 1} for j in range(8)
                ],
            }
        }

    # judge-folded honest ceiling driven DELIBERATELY below rho_lin (0.0 < 0.444) — the
    # exact smoke regime. It MUST be a recorded diagnostic, NOT the bracket point.
    entry = rel.compute_bracket(
        "broad_em",
        "betley",
        e0,
        context_ids,
        rho_lin=0.444,
        rng=rng,
        n_split_seeds=20,
        n_boot=200,
        judge_honest_sqrt=0.0,
    )

    bw = entry["bracket_width"]
    lo, hi = entry["bracket_width_ci"]
    # (a) bracket is ordered (floor <= ceiling).
    assert entry["bracket"][0] <= entry["bracket"][1], (
        f"bracket must be ordered, got {entry['bracket']}"
    )
    # (b) the width point estimate lies inside its own CI (the invariant assert).
    assert lo <= bw <= hi, f"bracket_width {bw} outside its CI [{lo}, {hi}]"
    # (c) no negative width with a positive `headroom` verdict (the impossible artifact).
    assert not (bw < 0 and entry["gate_verdict"] == "headroom"), (
        f"impossible output: negative bracket_width {bw} with headroom verdict"
    )
    # the headline is the binomial estimator; the judge-folded value is a diagnostic.
    assert entry["headline_estimator"] == "binomial_variance"
    assert entry["sqrt_r_yy_headline"] == entry["sqrt_r_yy_binomial"]
    assert entry["sqrt_r_yy_honest_judge_folded"] == 0.0  # recorded, not headlined
    # the headline point sits inside the bootstrap ceiling CI too (same-estimator).
    cl, ch = entry["sqrt_r_yy_ci"]
    assert cl <= entry["sqrt_r_yy_headline"] <= ch


# --------------------------------------------------------------------------- #
# 12. CONCERN judge-rerun-transport-undercount — the transport_deviation call    #
#     estimate loops over CONTEXTS (50/genre), so the default full-run estimate  #
#     is the registered ~16,000, NOT the context-omitting ~320.                  #
# --------------------------------------------------------------------------- #
def test_transport_deviation_full_run_estimate_is_16000():
    """The default full run (2 genres × 50 contexts × 4 behaviors × R=2 × J=20) must
    estimate 16,000 re-judge calls (plan §11 row 8), NOT the prior round's ~320 (which
    dropped the per-context loop)."""
    import issue742_judge_rerun as jr

    genres = ["betley", "ultrachat"]
    behaviors = list(jr.dc.READOUT_BEHAVIORS)
    counts = jr._resolve_context_counts(genres, None)
    # the canonical context count is 50/genre (or the real per-genre count when staged).
    n_contexts_total = sum(counts.values())
    n_cells = n_contexts_total * len(behaviors)
    est = n_cells * 2 * 20  # R_rerun=2, J=20
    assert est == 16000, (
        f"default full-run re-judge estimate must be the registered 16,000, got {est} "
        f"(counts={counts}); the context loop must be in the estimate"
    )
    # and max_contexts trims it (smoke parity): 1 context/genre -> tiny estimate.
    smoke_counts = jr._resolve_context_counts(genres, 1)
    assert sum(smoke_counts.values()) == 2, smoke_counts
