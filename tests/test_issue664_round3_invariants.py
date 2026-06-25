"""Issue #664 round-3 invariant pins (the reconciler-binding-FAIL fixes).

Pins the round-3 fixes so a future refactor cannot silently strip them
(the un-CI-pinned-assertion class). All CPU-only — they import the
``scripts/issue664_*`` modules and exercise the pure-Python topology / data
routing, stubbing the single HF-Hub / model touch points so no GPU / network
is required (the M7 equivalence test builds a tiny 2-layer CPU model).

Covered (the round-3 union list — the reconciler's binding FAIL):

- B6  sycophancy/refusal trained-store activations use the behavior's OWN
      battery (Sharma wrong-claims / #390 refusal-requests), NOT the generic
      48 preregistered Betley probes; em/bad_medical use the Betley battery;
      marker keeps the preregistered-48 surface. The store and the judged-rate
      eval score the SAME prompts per behavior (one source of truth).
- B7  the documented production launcher (--phase all --cells 1 --smoke
      --live-judge-smoke) reaches a CONTENT cell, so _live_judge_smoke calls
      judge_cell(..., live_judge=True) instead of returning N/A.
- M6  baseline propensity covers EVERY content behavior with a judged column
      (sycophancy/refusal + fact/em/bad_medical), not just sycophancy/refusal.
- M7  the completion log-prob batches >=2 (prompt, completion) pairs per HF
      forward (no batch-1 loop) AND the batched path matches the serial path
      (cosine >= 0.999 on a tiny CPU model under left-pad).
- N2  the registry manifest describes ONLY the selected cells, not the full
      realized grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue404_common as i4  # noqa: E402
import issue664_common as C  # noqa: E402
import issue664_dispatch as D  # noqa: E402
import issue664_eval as E  # noqa: E402

# ── B6: behavior-own store/eval batteries (the decisive defect) ───────────────
# NOTE (§16 strategy pivot): the trained-store + judged-rate batteries are now
# resolved through the ONE canonical resolver
# (``issue664_common.canonical_battery_for_*``). The B6 INTENT is unchanged --
# behavior-own batteries, store surface == eval surface -- but the routing goes
# through the single resolver, and the §16 contract CORRECTS two surfaces the r3
# router got wrong: ``marker`` now resolves to its #545 ``marker_eval_questions``
# battery (NOT the generic preregistered-48) and ``bad_medical`` to its #545
# ``fam_expr_bad_medical`` battery (NOT the Betley-8 placeholder). These tests
# were updated to the corrected canonical semantics; the references to the deleted
# ``S._behavior_battery`` / ``E._column_probes`` helpers re-point at the resolver.


def _beh_qs(behavior: str) -> list[str]:
    """Flat questions the store path reads for a behavior (canonical resolver)."""
    return [it["question"] for it in C.canonical_battery_for_behavior(behavior)]


def _col_qs(column: str) -> list[str]:
    """Flat questions the eval path reads for a column (canonical resolver)."""
    return [it["question"] for it in C.canonical_battery_for_column(column)]


def test_b6_sycophancy_store_battery_is_wrong_claims_not_preregistered() -> None:
    """The reconciler's DECISIVE finding: sycophancy store activations must be
    over the Sharma wrong-claim user turns (the SAME surface the training
    positives + judge use), NOT i4.fetch_preregistered_probes(48). Under §16
    sycophancy is an explicit ISSUE_664_BATTERY_OVERRIDES entry."""
    syco = _beh_qs("sycophancy")
    pre48 = set(i4.fetch_preregistered_probes(48))
    bet8 = set(i4.fetch_betley_main_8())
    assert "sycophancy" in C.ISSUE_664_BATTERY_OVERRIDES
    assert len(syco) == 50  # the 50-claim wrong-claim read
    assert set(syco) != pre48, "B6 regression: sycophancy store uses the generic 48 probes"
    assert set(syco) != bet8, "B6 regression: sycophancy store uses the Betley-8 placeholder"
    # the store surface is IDENTICAL to the judged-rate eval surface for syco.
    assert set(syco) == set(_col_qs("sycophancy"))


def test_b6_refusal_store_battery_is_request_pool_not_preregistered() -> None:
    refu = _beh_qs("refusal")
    pre48 = set(i4.fetch_preregistered_probes(48))
    bet8 = set(i4.fetch_betley_main_8())
    assert "refusal" in C.ISSUE_664_BATTERY_OVERRIDES
    assert refu, "refusal battery empty"
    assert set(refu) != pre48, "B6 regression: refusal store uses the generic 48 probes"
    assert set(refu) != bet8, "B6 regression: refusal store uses the Betley-8 placeholder"
    # store == eval surface; both the #390 refusal-request pool.
    assert set(refu) == set(_col_qs("refusal"))
    assert set(refu) == set(C.refusal_request_pool())


def test_b6_em_and_bad_medical_use_own_545_battery_not_preregistered() -> None:
    """em -> broad_em (Betley-main-8, content-identical to fetch_betley_main_8);
    bad_medical -> its OWN #545 fam_expr_bad_medical battery (the §16 correction:
    the r3 router wrongly pinned bad_medical to Betley-8). Neither is the generic
    preregistered-48."""
    bet8 = set(i4.fetch_betley_main_8())
    pre48 = set(i4.fetch_preregistered_probes(48))
    # em routes via broad_em -> betley_main8.json (same content as the helper).
    em = set(_beh_qs("em"))
    assert em == bet8, "em store should be the broad_em / Betley main-8 battery"
    assert em != pre48, "em store must NOT be the generic 48 probes"
    # bad_medical routes via its OWN #545 column battery, NOT Betley-8.
    bm = set(_beh_qs("bad_medical"))
    assert C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN["bad_medical"] == "fam_expr_bad_medical"
    assert bm == set(_col_qs("fam_expr_bad_medical"))
    assert bm != bet8, "bad_medical should resolve to its OWN #545 battery, NOT Betley-8"
    assert bm != pre48, "bad_medical store must NOT be the generic 48 probes"
    # ic_edu maps to the em (broad_em) surface; tf_rev to the fact surface.
    assert set(_beh_qs("ic_edu")) == bet8
    assert set(_beh_qs("tf_rev")) == set(_beh_qs("fact"))


def test_b6_marker_resolves_to_own_545_battery() -> None:
    """§16 correction: marker resolves to its #545 ``marker_eval_questions``
    column battery (each column self-routes via its own ColumnSpec.battery), NOT
    a hand-rolled fetch_preregistered_probes(48) special case. The r3 router
    pinned marker to the generic 48 probes -- the pivot fixes that."""
    from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS

    assert "marker" not in C.ISSUE_664_BATTERY_OVERRIDES
    assert COLUMNS["marker"].battery == "marker_eval_questions.json"
    mark = set(_beh_qs("marker"))
    assert mark == set(_col_qs("marker"))
    assert mark != set(i4.fetch_preregistered_probes(48)), (
        "marker must resolve to its own #545 battery, NOT the generic preregistered-48"
    )


def test_b6_store_battery_equals_eval_column_per_content_behavior() -> None:
    """One source of truth: the trained-store activation surface == the judged-
    rate eval surface for EVERY content behavior (so the gate DV's activations
    and the judge labels score the same prompts). Both call the ONE resolver."""
    for behavior in C.CONTENT_BEHAVIORS:
        column = C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[behavior]
        assert set(_beh_qs(behavior)) == set(_col_qs(column)), (
            f"store/eval surface mismatch for {behavior}"
        )


# ── B7: live-judge-smoke reaches the production judge branch ───────────────────
def test_b7_live_judge_smoke_selection_includes_a_content_cell() -> None:
    """The documented `--phase all --cells 1 --smoke --live-judge-smoke` must
    select at least one CONTENT cell so _live_judge_smoke reaches judge_cell(...,
    live_judge=True) instead of returning 'B5 slice N/A' (the prior topology bug:
    marker-only canary, marker ∉ CONTENT_BEHAVIORS)."""

    class Args:
        smoke = True
        cells = 1
        live_judge_smoke = True
        gpu_id = 0
        phase = "all"

    sel = D._select_cells(Args())
    assert any(c.behavior in C.CONTENT_BEHAVIORS for c in sel), (
        "B7 regression: no content-behavior cell in the live-judge smoke selection"
    )
    # the marker band-stop architecture canary is still kept too.
    assert any(c.behavior == "marker" for c in sel)


def test_b7_non_live_judge_smoke_stays_marker_only() -> None:
    class Args:
        smoke = True
        cells = 1
        live_judge_smoke = False
        gpu_id = 0
        phase = "all"

    sel = D._select_cells(Args())
    assert len(sel) == 1 and sel[0].behavior == "marker"


def test_b7_live_judge_smoke_reaches_judge_cell_live_true(monkeypatch, tmp_path) -> None:
    """End-to-end TOPOLOGY pin: the launcher's _live_judge_smoke path calls
    judge_cell(..., live_judge=True) on a content cell. We stub judge_cell so no
    GPU/API is needed and assert it is invoked with live_judge=True + writes a
    real all_scores-shaped artifact the live-judge gate reads."""

    class Args:
        smoke = True
        cells = 1
        live_judge_smoke = True
        gpu_id = 0
        phase = "all"

    sel = D._select_cells(Args())
    captured = {}

    def _fake_judge_cell(cell, *, smoke, live_judge=False):
        captured["cell"] = cell
        captured["smoke"] = smoke
        captured["live_judge"] = live_judge
        out = tmp_path / "judged_rates.json"
        out.write_text(json.dumps({"rates": {"sycophancy__ctx": {"n_judged": 3, "rate": 0.5}}}))
        return out

    import issue664_eval as Emod

    monkeypatch.setattr(Emod, "judge_cell", _fake_judge_cell)
    monkeypatch.setattr(D, "phase_log", lambda *a, **k: None)
    D._live_judge_smoke(sel)
    assert captured.get("live_judge") is True, "B7: judge_cell not called with live_judge=True"
    assert captured["cell"].behavior in C.CONTENT_BEHAVIORS


# ── M6: baseline propensity covers ALL content behaviors ──────────────────────
def test_m6_baseline_probe_pool_covers_all_content_behaviors(monkeypatch) -> None:
    """Every content behavior has a non-empty baseline probe pool (the prior
    code returned [] for fact/em/bad_medical). Stub the HF wrong-claims download
    so no network is needed; the non-HF behaviors resolve locally."""

    # stub the canonical column resolver (HF-backed for the override columns) so
    # no network is needed; every column resolves to a 40-probe-item list.
    monkeypatch.setattr(
        C,
        "canonical_battery_for_column",
        lambda column, *, smoke=False: [
            {"probe_id": f"{column}_{i}", "question": f"{column}_probe_{i}"} for i in range(40)
        ],
    )
    for behavior in C.CONTENT_BEHAVIORS:
        pool = D._baseline_probe_pool(behavior, smoke=False)
        assert pool, f"M6 regression: empty baseline probe pool for {behavior}"
        assert len(pool) <= 30  # capped baseline read


def test_m6_rated_behaviors_are_all_content_behaviors_in_grid() -> None:
    """The `rated` list in _write_baseline_propensity is exactly the content
    behaviors present (not the hardcoded {sycophancy, refusal}). We pin the
    selection logic that drives it."""
    grid = C.realized_grid()
    behaviors = sorted({c.behavior for c in grid})
    rated = [b for b in behaviors if b in C.CONTENT_BEHAVIORS]
    # all five content behaviors are realized in the #664 grid.
    assert set(rated) == set(C.CONTENT_BEHAVIORS)
    # the marker carve-out: marker is NOT rated (no bare-context judged rate).
    assert "marker" not in rated
    # every rated behavior maps to a real judge column.
    for b in rated:
        col = C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[b]
        assert col, f"{b} has no primary registry column for the baseline judge"


# ── M7: batched completion log-prob (>=2 pairs/forward + serial equivalence) ──
def _tiny_qwen_cpu():
    """A tiny 2-layer Qwen2 CausalLM on CPU sharing the real Qwen-2.5 vocab so
    the real tokenizer's ids index a valid embedding. Random init is fine — the
    equivalence test compares the batched vs serial path on the SAME weights."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(C.QWEN_ID, trust_remote_code=True)
    cfg = Qwen2Config(
        vocab_size=tok.vocab_size + len(tok.get_added_vocab()) + 8,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).eval().to(torch.float32)
    return model, tok


def _serial_lennorm_logp(model, tok, context_id, pairs):
    """The PRE-fix serial reference: one batch-1 forward per pair, no padding."""
    import torch
    import torch.nn.functional as torch_f

    out = []
    for q, comp in pairs:
        prompt = E._prompt_text_for(context_id, q)
        p_ids = tok.encode(prompt, add_special_tokens=False)
        c_ids = tok.encode(comp, add_special_tokens=False)
        if not c_ids:
            out.append(None)
            continue
        ids = torch.tensor([p_ids + c_ids])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]
        logp = 0.0
        for t, tok_id in enumerate(c_ids):
            pos = len(p_ids) + t - 1
            logp += torch_f.log_softmax(logits[pos].float(), dim=-1)[tok_id].item()
        out.append(logp / len(c_ids))
    return out


def test_m7_batched_logp_matches_serial_under_left_pad(monkeypatch) -> None:
    """The batched rewrite must equal the serial batch-1 path (cosine >= 0.999)
    on >=2 pairs of DIFFERENT lengths (so left-padding actually fires). Guards
    the left-pad position_ids / per-row pad-offset correctness (#502 trap)."""
    import math

    import numpy as np
    import torch

    model, tok = _tiny_qwen_cpu()

    # monkeypatch the model loader + prompt builder so _lennorm_logp uses our
    # tiny CPU model + a trivial deterministic prompt (no chat template / HF).
    monkeypatch.setattr(E, "_prompt_text_for", lambda ctx, q: f"CTX:{ctx} Q:{q} A:")

    class _Loader:
        @staticmethod
        def from_pretrained(*a, **k):
            return model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Loader)
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: tok)}),
    )

    # DIFFERENT-length completions so left-pad widths differ across the batch.
    pairs = [
        ("What is two plus two?", "The answer is four."),
        ("Name a color.", "Blue is a common primary color that appears in the sky."),
        ("", "empty-prompt completion"),
        ("Another question here?", ""),  # empty completion -> None
    ]

    serial = _serial_lennorm_logp(model, tok, "ctx0", pairs)
    batched = E._lennorm_logp("ignored-path", "ctx0", pairs)

    assert len(batched) == len(pairs)
    # the empty-completion pair is None in BOTH paths.
    assert serial[3] is None and batched[3] is None
    # collect the non-None aligned values and compare (cosine + abs).
    s = np.array([v for v in serial if v is not None], dtype=np.float64)
    b = np.array([batched[i] for i, v in enumerate(serial) if v is not None], dtype=np.float64)
    assert s.shape == b.shape and s.size >= 2
    cos = float(np.dot(s, b) / (np.linalg.norm(s) * np.linalg.norm(b) + 1e-12))
    assert cos >= 0.999, f"M7 batched-vs-serial cosine {cos:.6f} < 0.999 (left-pad divergence?)"
    for sv, bv in zip(s, b, strict=True):
        assert math.isclose(sv, bv, rel_tol=1e-4, abs_tol=1e-4), f"per-pair logp drift {sv} vs {bv}"


def test_m7_processes_multiple_pairs_in_one_forward(monkeypatch) -> None:
    """Pin that the batched path issues a forward with batch_size > 1 (no batch-1
    loop). We capture the input_ids shape passed to the model."""
    import torch

    model, tok = _tiny_qwen_cpu()
    seen_shapes = []
    real_forward = model.forward

    def _spy_forward(*args, **kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        seen_shapes.append(tuple(ids.shape))
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", _spy_forward)
    monkeypatch.setattr(E, "_prompt_text_for", lambda ctx, q: f"Q:{q} A:")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        type("L", (), {"from_pretrained": staticmethod(lambda *a, **k: model)}),
    )
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: tok)}),
    )

    pairs = [(f"q{i}", f"completion number {i} text") for i in range(4)]
    E._lennorm_logp("ignored", "ctx0", pairs)
    assert seen_shapes, "no forward was issued"
    # at least one forward processed > 1 pair (batch dim > 1).
    assert any(shape[0] > 1 for shape in seen_shapes), (
        f"M7 regression: all forwards were batch-1 ({seen_shapes})"
    )


# ── N2: manifest describes ONLY the selected cells ────────────────────────────
def test_n2_manifest_uses_selected_cells_not_full_grid(tmp_path, monkeypatch) -> None:
    """write_manifest(cells) must enumerate ONLY the passed cells; the worker
    previously wrote the full realized_grid() regardless of --cells. We call the
    eval manifest writer directly with a 2-cell subset and confirm the manifest
    cell set matches."""
    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    grid = C.realized_grid()
    subset = grid[:2]
    out = E.write_manifest(subset, smoke=False)
    manifest = json.loads(out.read_text())
    cells_in_manifest = {row["cell"] for row in manifest["tuples"]}
    assert cells_in_manifest == {c.eval_key for c in subset}, (
        "N2 regression: manifest cell set != the selected subset"
    )
    assert manifest["n_cells"] == 2
    # sanity: the full grid has MORE than 2 cells, so the subset is a real filter.
    assert len(grid) > 2
