"""Round-2 regression tests for issue #665 — the six binding code-review blockers.

Each test fails pre-fix and passes post-fix:

- Blocker 1: the A3.6c parity gate HALTS (no model load / no forward pass) when the
  parity probe did NOT PASS — `_run_patch_gpu` must never be reached.
- Blocker 2: the parity probe + A3.6c patch use the REAL #664 context prompt
  (`C.context_chat_messages`), NOT a synthetic "Hello." — the literal is gone from
  the live GPU paths.
- Blocker 3: the aggregate computes BOTH `A3_10_rho_raw` AND
  `A3_10_rho_partial_E0_wnorm`, and REJECTS with a clear error when the
  per-cell g0_E0 inputs are absent.
- Blocker 4: judge_E persists the `judge_positive_rate` (PRIMARY DV) per context.
- Blocker 5: A3.9 sweeps the 4-key grid {c_C, ψ(t), ψ(δ), c_C+ψ(δ)}.
- Blocker 6: the aggregate carries family-clustered CIs + probe-split fields.

CPU-only: no HF / network / GPU. The GPU-bound paths are exercised via
monkeypatched side-effect trackers + synthetic per-cell JSON fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ── Blocker 1: parity gate HALTS before any model load ────────────────────────


def test_parity_fail_halts_before_any_model_load(tmp_path, monkeypatch):
    """A FAILing parity gate writes a SUPPRESSED record and CONTINUES *without*
    ever calling `_run_patch_gpu` (no model load, no forward pass). Run-and-flag
    would have called it; HALT-before-spend does not."""
    import issue665_common as C
    import issue665_patch_gpu as P

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(P, "A36C_DIR", tmp_path / "a36c")
    monkeypatch.setattr(P, "FITNESS_DIR", tmp_path / "adapter_fitness")

    # the parity gate reads no PASS -> FAIL
    monkeypatch.setattr(P, "parity_gate_passed", lambda cell: (False, "no parity_probe PASS"))

    # tripwire: _run_patch_gpu MUST NOT be reached on a parity FAIL.
    called = {"run": False, "load": False}

    def _boom_run(*a, **k):
        called["run"] = True
        raise AssertionError("_run_patch_gpu reached despite parity FAIL (no HALT)")

    monkeypatch.setattr(P, "_run_patch_gpu", _boom_run)

    # load_cell tripwire — must also never be reached (the gate is FIRST)
    import explore_persona_space.analysis.gate_io as gate_io

    def _boom_load(*a, **k):
        called["load"] = True
        raise AssertionError("load_cell reached despite parity FAIL (no HALT)")

    monkeypatch.setattr(gate_io, "load_cell", _boom_load)

    monkeypatch.setattr(sys, "argv", ["issue665_patch_gpu.py", "--cells", C.A36C_SUBSET[0]])
    P.main()

    assert called["run"] is False, "parity FAIL must HALT before _run_patch_gpu"
    assert called["load"] is False, "parity FAIL must HALT before load_cell"

    # the suppressed record is backward-compatible with the aggregate/figures pipeline
    rec_path = tmp_path / "a36c" / f"{C.A36C_SUBSET[0]}.json"
    assert rec_path.exists(), "a suppressed record must still be written"
    rec = json.loads(rec_path.read_text())
    assert rec["skipped"] is True
    assert rec["f_CV"] is None
    assert rec["trusted"] is False
    assert rec["rows"] == []


def test_parity_pass_runs_patch(tmp_path, monkeypatch):
    """The guard is not a false-positive: a PASSing parity gate DOES reach
    `_run_patch_gpu` (the cell is patched)."""
    import issue665_common as C
    import issue665_patch_gpu as P

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(P, "A36C_DIR", tmp_path / "a36c")
    monkeypatch.setattr(P, "FITNESS_DIR", tmp_path / "adapter_fitness")
    monkeypatch.setattr(P, "parity_gate_passed", lambda cell: (True, "parity PASS"))

    ran = {"run": False}

    def _fake_run(cell, contexts, layers, scopes):
        ran["run"] = True
        return {"rows": [{"context_id": "src", "f_cv_v": 1.0, "e_read": 1.0}]}

    monkeypatch.setattr(P, "_run_patch_gpu", _fake_run)

    # load_cell stub so the bystander selection runs without HF
    import explore_persona_space.analysis.gate_io as gate_io

    class _SC:
        tensors = {"context_ids": ["src", "b1", "b2"]}
        source_ctx_id = "src"

        def free(self):
            pass

    monkeypatch.setattr(gate_io, "load_cell", lambda *a, **k: _SC())
    monkeypatch.setattr(sys, "argv", ["issue665_patch_gpu.py", "--cells", C.A36C_SUBSET[0]])
    P.main()
    assert ran["run"] is True


# ── Blocker 2: real #664 context prompts (no synthetic "Hello.") ──────────────


def test_no_hardcoded_hello_in_live_gpu_paths():
    """The live GPU paths build c_C on the REAL #664 prompt via
    C.context_chat_messages, NOT a synthetic 'Hello.'. The msgs=[{...:'Hello.'}]
    pattern must be gone from the executable code of both GPU scripts."""
    for fname in ("issue665_patch_gpu.py", "issue665_parity_probe.py"):
        src = (_SCRIPTS / fname).read_text()
        # the banned construction (a chat message literally content 'Hello.')
        assert '"content": "Hello."' not in src, f"{fname} still builds a synthetic Hello. prompt"
        assert "context_chat_messages" in src, f"{fname} must use the real #664 context prompt"


def test_context_chat_messages_builds_real_prompt(monkeypatch):
    """C.context_chat_messages resolves the #594 battery instance for a context id
    and routes through issue664_common.context_messages (the #664 c_C recipe)."""
    import issue664_common
    import issue665_common as C

    fake_inst = {
        "id": "f1_house_librarian",
        "system_prompt": "You are a librarian.",
        "family": "persona",
        "prefix_messages": [],
    }
    monkeypatch.setattr(C, "_BATTERY_CACHE", {"f1_house_librarian": fake_inst})
    seen = {}

    def _cm(inst, q):
        seen["inst"] = inst
        seen["q"] = q
        return [
            {"role": "system", "content": inst["system_prompt"]},
            {"role": "user", "content": q},
        ]

    monkeypatch.setattr(issue664_common, "context_messages", _cm)
    msgs = C.context_chat_messages("f1_house_librarian", "What is the capital of France?")
    assert seen["inst"] is fake_inst
    assert seen["q"] == "What is the capital of France?"
    assert msgs[0]["content"] == "You are a librarian."
    # an unknown context id fails loud (can't rebuild the real prompt)
    with pytest.raises(ValueError, match="not found in the #594 battery"):
        C.context_chat_messages("nonexistent_ctx", "q")


# ── Blocker 3: A3.10 E0/‖ŵ‖ partial (raw + partial; reject when absent) ───────


def test_a310_partial_rejects_when_inputs_absent():
    """The E0/‖ŵ‖ partial REJECTS with a clear error when no g0_E0.json present."""
    import issue665_aggregate as A

    with pytest.raises(ValueError, match=r"E0.*partial|g0_E0\.json"):
        A._a310_partial_E0_wnorm(["bm_default_contra_d1_seed42"], {})


def test_a310_partial_computes_raw_and_partial():
    """With pooled (g0, ghat, E0, ‖ŵ‖) the aggregate returns BOTH the raw Spearman
    and the partial with E0+‖ŵ‖ partialled out (Blocker 3c)."""
    import issue665_aggregate as A

    rng = np.random.default_rng(0)
    cells = ["bm_default_contra_d1_seed42", "bm_librarian_contra_d1_seed42"]
    g0_e0 = {}
    for ci, cell in enumerate(cells):
        entries = []
        for k in range(25):
            e0 = float(rng.standard_normal())
            g0 = float(rng.standard_normal())
            # ghat correlated with g0 + E0 so the partial differs from the raw
            ghat = 0.6 * g0 + 0.5 * e0 + 0.3 * float(rng.standard_normal())
            entries.append(
                {"context_id": f"f{k}_c", "g0": g0, "ghat_real": ghat, "E0": e0, "wnorm": 3.0 + ci}
            )  # ‖ŵ‖ varies across cells
        g0_e0[cell] = {"entries": entries}
    res = A._a310_partial_E0_wnorm(cells, g0_e0)
    assert "A3_10_rho_raw" in res and res["A3_10_rho_raw"] is not None
    assert "A3_10_rho_partial_E0_wnorm" in res and res["A3_10_rho_partial_E0_wnorm"] is not None
    assert res["wnorm_partialled"] is True  # ‖ŵ‖ varied across the two cells
    assert np.isfinite(res["A3_10_rho_raw"])
    assert np.isfinite(res["A3_10_rho_partial_E0_wnorm"])
    # partialling E0 out should change the correlation (E0 is a real confounder here)
    assert abs(res["A3_10_rho_raw"] - res["A3_10_rho_partial_E0_wnorm"]) > 1e-6


# ── Blocker 4: judge_E persists the judge-positive RATE ───────────────────────


def test_judge_E_persists_positive_rate(tmp_path):
    """`_per_context_completion_scores` re-reads save_raw and groups per-completion
    scores back to each context so the per-context judge-positive RATE (PRIMARY DV)
    is derivable — not just the mean."""
    import issue665_judge_E as JE

    raw = {
        "all_scores": {
            "f6_helpful_asst__00000__00": {"aligned": 80, "coherent": 90},
            "f6_helpful_asst__00001__00": {"aligned": 20, "coherent": 90},
            "f1_house_librarian__00002__00": {"aligned": 60, "coherent": 90},
        }
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw))
    out = JE._per_context_completion_scores(raw_path, ["f6_helpful_asst", "f1_house_librarian"])
    assert out["f6_helpful_asst"] == [80.0, 20.0]
    assert out["f1_house_librarian"] == [60.0]
    # the rate is fraction >= threshold (50): helpful_asst 1/2, librarian 1/1
    helpful = out["f6_helpful_asst"]
    rate = sum(1 for s in helpful if s >= JE.JUDGE_THRESHOLD) / len(helpful)
    assert rate == 0.5


# ── Blocker 5: A3.9 4-key grid ────────────────────────────────────────────────


def test_a39_sweeps_four_keys():
    """arm_a39 sweeps {c_C, ψ(t), ψ(δ), c_C+ψ(δ)} x {I, diag, Σinv} (Blocker 5)."""
    import issue665_gate_cpu as G

    from explore_persona_space.analysis.whitened_gate import METRIC_KEYS

    assert G.A39_KEY_LABELS == ("c_C", "psi_t", "psi_delta", "c_C_plus_psi_delta")

    # synthetic StoreCell-like object with the tensors arm_a39 reads
    import torch

    d, n_ctx, n_layer = 16, 6, 4
    rng = np.random.default_rng(1)

    class _SC:
        source_idx = 0
        tensors = {
            "v_plus": torch.tensor(rng.standard_normal((n_ctx, n_layer, d)), dtype=torch.float32),
            "v0": torch.tensor(rng.standard_normal((n_ctx, n_layer, d)), dtype=torch.float32),
            "c_C_base": torch.tensor(rng.standard_normal((n_ctx, n_layer, d)), dtype=torch.float32),
            "c_C_trained": torch.tensor(
                rng.standard_normal((n_ctx, n_layer, d)), dtype=torch.float32
            ),
            "t_CB": torch.tensor(rng.standard_normal((n_layer, d)), dtype=torch.float32),
        }

    sigma_c_layer = np.eye(d) + 0.1 * rng.standard_normal((d, d))
    sigma_c_layer = sigma_c_layer @ sigma_c_layer.T  # PSD
    out = G.arm_a39(_SC(), layer=1, lam=1e-2, sigma_c_layer=sigma_c_layer)
    cells = out["key_metric_results"]
    # all 4 keys x 3 metrics present
    for k in G.A39_KEY_LABELS:
        for m in METRIC_KEYS:
            assert f"{k}::{m}" in cells, f"missing A3.9 cell {k}::{m}"
            assert cells[f"{k}::{m}"]["key"] == k
            assert cells[f"{k}::{m}"]["metric"] == m
    assert len(cells) == 4 * len(METRIC_KEYS)
    # the boxed verdict is c_C key + Sigma_inv metric
    assert "verdict_ii_sigma_inv_wins" in out
    assert "best_key_metric" in out


# ── Blocker 6: family clustering + probe-split replication fields ──────────────


def test_family_of_context_hierarchical():
    """Blocker 6a: the hierarchical family grain comes from the #594 f1..f8 prefix,
    NOT the bare source_seed."""
    import issue665_common as C

    assert C.family_of_context("f1_house_librarian") == "persona"
    assert C.family_of_context("f6_helpful_asst") == "default"
    assert C.family_of_context("f2_wildchat_001") == "wildchat"
    assert C.family_of_context("unknown_ctx") == "other"


def test_aggregate_carries_family_and_probe_split_fields(tmp_path, monkeypatch):
    """The aggregate per_behavior block carries family_clustered_ci_*, the
    probe_split_* fields (count == 8), AND the A3.10 raw/partial (Blockers 3c+6)."""
    import issue665_aggregate as A
    import issue665_common as C

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    cells = ["bm_default_contra_d1_seed42", "bm_librarian_contra_d1_seed42"]

    # synthesize the a310 / a39 / a38 per-cell arm JSONs + the per-cell g0_E0.json
    rng = np.random.default_rng(2)
    for cell in cells:
        layer = str(C.read_layer_for_cell(cell))
        (tmp_path / "a310").mkdir(parents=True, exist_ok=True)
        (tmp_path / "a39").mkdir(parents=True, exist_ok=True)
        (tmp_path / "a38").mkdir(parents=True, exist_ok=True)
        (tmp_path / "a310" / f"{cell}.json").write_text(
            json.dumps({"by_layer": {layer: {"g0_spearman": 0.4, "gplus_spearman": 0.6}}})
        )
        (tmp_path / "a39" / f"{cell}.json").write_text(
            json.dumps(
                {
                    "by_layer": {
                        layer: {
                            "cosine_spearman": 0.2,
                            "verdict_i_some_beats_cosine": True,
                            "verdict_ii_sigma_inv_wins": True,
                        }
                    }
                }
            )
        )
        (tmp_path / "a38" / f"{cell}.json").write_text(
            json.dumps(
                {"by_layer": {layer: {"median_rankone_residual": 0.1, "svd_sigma1_frac": 0.8}}}
            )
        )
        # the g0_E0 per-cell file (Blocker 3a)
        (tmp_path / "per_cell" / cell).mkdir(parents=True, exist_ok=True)
        entries = []
        for k in range(20):
            e0 = float(rng.standard_normal())
            g0 = float(rng.standard_normal())
            entries.append(
                {
                    "context_id": f"f{(k % 8) + 1}_c{k}",
                    "g0": g0,
                    "ghat_real": 0.5 * g0 + 0.4 * e0,
                    "E0": e0,
                    "wnorm": 3.0,
                }
            )
        (tmp_path / "per_cell" / cell / "g0_E0.json").write_text(json.dumps({"entries": entries}))

    # stub the probe-split path (it loads tensors from HF; replace with a fixed floor)
    monkeypatch.setattr(
        A,
        "_probe_split_replication",
        lambda beh_cells, rng: {
            "probe_split_floor_mean": 0.05,
            "probe_split_floor_ci_lo": 0.02,
            "probe_split_floor_ci_hi": 0.08,
            "probe_split_replication_count": C.PROBE_SPLIT_R,
            "n_cells": len(beh_cells),
        },
    )

    agg = A.aggregate(cells, smoke=True)
    pb = agg["per_behavior"]["bad_medical"]
    # Blocker 6c: family-clustered CI present
    assert "family_clustered_ci_g0_spearman" in pb
    assert pb["family_clustered_ci_g0_spearman"]["mean"] is not None
    # Blocker 6b: probe-split fields present, replication count == 8
    assert pb["probe_split_floor_mean"] is not None
    assert pb["probe_split_replication_count"] == 8
    # Blocker 3c: raw + partial present
    assert "A3_10_rho_raw" in pb and pb["A3_10_rho_raw"] is not None
    assert "A3_10_rho_partial_E0_wnorm" in pb and pb["A3_10_rho_partial_E0_wnorm"] is not None
