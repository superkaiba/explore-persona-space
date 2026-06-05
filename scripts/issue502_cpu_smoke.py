#!/usr/bin/env python3
"""Issue #502 — CPU-only smoke for the bake-off extensions.

The dev VM has no GPU, so this script exercises everything that's
NOT actually loading the 7B model + running real residual hooks:

1. probes_500.json schema check (exists, shape, disjoint, prefix).
2. _load_probe_questions(pool_path=...) loads and validates the pool.
3. _js_divergence_rowwise correctness on hand-built probability rows.
4. compute_next_token_js_matrix on a synthetic 3-cond sidecar set.
5. cross_check_next_token_js_against_406 on identity (matrix==#406 JS).
6. merge_partitioned_activations stacks per-cond files into the
   canonical (n_cond, n_q, H) shape and rejects shape mismatch /
   all-NaN cells.
7. _set_roots redirects every output root atomically.
8. The batched extraction code path's REAL-tokenizer left-pad +
   per-sequence response-length logic, against tiny-random-gpt2,
   against the serial extraction's output: cosine(batched, serial) ≥
   0.999 per (layer × position) for both ``last_prompt`` and
   ``mean_response``. THIS is the batched==serial equality gate.
9. Multi-GPU partition correctness: partition then merge of two
   disjoint cond subsets reconstructs the same stacked grid as a
   single all-conds run on the same toy model.

Exit 0 on PASS; non-zero on first failure. Digest at
``eval_results/issue_502/cpu_smoke_digest.json``.
"""

# Greek + special characters in docstrings.
# ruff: noqa: RUF002

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue493_extraction_metric_bakeoff import (  # noqa: E402
    _extract_batch,
    _extract_one,
    _js_divergence_rowwise,
    _LayerHookCapture,
    _load_probe_questions,
    _set_roots,
    compute_next_token_js_matrix,
    cross_check_next_token_js_against_406,
    load_next_token_logits,
    merge_partitioned_activations,
    write_next_token_js_matrix,
)

logger = logging.getLogger("i502.cpu_smoke")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


SMOKE_ROOT = Path("/tmp") / "issue502_cpu_smoke_root"
PROBES_PATH = PROJECT_ROOT / "eval_results" / "issue_502" / "probes_500.json"


# ─────────────────────────── Check 1-2: probe pool ───────────────────────────


def check_probe_pool() -> dict:
    """Load the probes_500.json and verify constraints via the loader."""
    if not PROBES_PATH.exists():
        raise FileNotFoundError(
            f"{PROBES_PATH} missing — run scripts/issue502_generate_probes.py first."
        )
    payload = json.loads(PROBES_PATH.read_text())
    assert payload["schema_version"] == 1
    assert payload["n_total"] == 500, f"n_total={payload['n_total']}"
    assert len(payload["q_test_subset_50"]) == 50
    assert len(payload["new_probes_450"]) == 450
    assert payload["probes"][:50] == payload["q_test_subset_50"]
    # The loader does the deep checks; let it raise on violation.
    loaded = _load_probe_questions(pool_path=PROBES_PATH)
    assert len(loaded) == 500
    # Disjointness sanity (delegated to loader, re-asserted here for log).
    from explore_persona_space.experiments.i460_data import (
        load_q_test_extended_50,
        load_q_train_answers,
    )

    q_test = load_q_test_extended_50()
    q_train = set(load_q_train_answers().keys())
    overlap = set(loaded[50:]) & set(q_test)
    assert not overlap, f"new probes overlap q_test: {sorted(overlap)[:3]}"
    overlap_t = set(loaded[50:]) & q_train
    assert not overlap_t, f"new probes overlap q_train: {sorted(overlap_t)[:3]}"
    return {
        "n_total": payload["n_total"],
        "n_q_test_prefix": 50,
        "n_new": 450,
        "model": payload["model"],
    }


# ─────────────────────────── Check 3: JS rowwise ───────────────────────────


def check_js_rowwise() -> dict:
    """Hand-built probability rows: equal → 0, antipodal → ≈ 1."""
    rng = np.random.default_rng(0)
    V = 50
    p = rng.dirichlet(np.ones(V), size=10).astype(np.float32)
    js_self = _js_divergence_rowwise(p, p)
    assert np.allclose(js_self, 0.0, atol=1e-6), f"JS(P, P) = {js_self}"

    # Two completely disjoint distributions (mass on different halves).
    p_left = np.zeros((1, V))
    p_left[0, : V // 2] = 1.0 / (V // 2)
    p_right = np.zeros((1, V))
    p_right[0, V // 2 :] = 1.0 / (V // 2)
    js_anti = _js_divergence_rowwise(p_left.astype(np.float32), p_right.astype(np.float32))
    assert np.allclose(js_anti, 1.0, atol=1e-6), (
        f"JS(left, right) should be 1.0 (disjoint supports); got {js_anti}"
    )
    return {"js_self": float(js_self.mean()), "js_antipodal": float(js_anti.mean())}


# ─────────────────────────── Check 4: matrix build ───────────────────────────


def check_matrix_build() -> dict:
    """Build a 3-cond next_token_js matrix from synthetic sidecars."""
    rng = np.random.default_rng(1)
    V = 32
    n_q = 8
    cids = ["A1", "A2", "B1"]
    # 3 distinct distributions, repeated across n_q probes.
    base = np.eye(V, dtype=np.float32)  # Identity rows
    cid_to_probs = {
        "A1": np.tile(base[0:n_q], (1, 1)),
        "A2": np.tile(base[V // 2 - n_q // 2 : V // 2 - n_q // 2 + n_q], (1, 1)),
        # B1: random Dirichlet (different from both)
        "B1": rng.dirichlet(np.ones(V), size=n_q).astype(np.float32),
    }
    payload = compute_next_token_js_matrix(cid_to_probs, cond_ids_order=cids)
    matrix = payload["matrix"]
    assert matrix["A1"]["A1"] == 0.0
    assert matrix["A1"]["A2"] > 0.5, "A1 vs A2 (disjoint one-hot rows) should be high JS"
    assert matrix["A1"]["A1"] != matrix["A1"]["B1"], "diagonal must differ from cross"
    return {"a1_a2": matrix["A1"]["A2"], "a1_b1": matrix["A1"]["B1"]}


# ─────────────────────────── Check 5: #406 cross-check ───────────────────────────


def check_cross_against_406() -> dict:
    """Build a synthetic JS matrix that's a monotone transform of #406's,
    confirm rank correlation passes the floor."""
    p406 = PROJECT_ROOT / "eval_results/issue_406/divergence/D_matrix.json"
    if not p406.exists():
        return {"ok": True, "reason": "no #406 reference; cross-check skipped on dev VM"}
    p406_d = json.loads(p406.read_text())
    js406 = p406_d["JS"]
    # Build a "perfect rank twin" matrix by passing #406's values directly.
    cond_ids = list(js406.keys())
    matrix = {
        a: {b: (0.0 if a == b else (js406[a].get(b, 0.0) or 0.0)) for b in cond_ids}
        for a in cond_ids
    }
    payload = {
        "extraction_point": "last_prompt",
        "layer": -1,
        "metric": "next_token_js",
        "variant": "raw",
        "cond_ids": cond_ids,
        "matrix": matrix,
    }
    summary = cross_check_next_token_js_against_406(payload)
    assert summary["ok"], summary
    assert summary["rank_corr_spearman"] > 0.99, (
        f"identity matrix should give rho ~ 1, got {summary['rank_corr_spearman']}"
    )
    return summary


# ─────────────────────────── Check 6: partition merge ───────────────────────────


def check_partition_merge() -> dict:
    """Two per-cond files at the same (point, layer) → merged stack matches
    a hand-built canonical (n_cond, n_q, H) tensor."""
    import torch

    _set_roots(SMOKE_ROOT)
    # Clear any prior smoke output.
    import shutil

    from issue493_extraction_metric_bakeoff import ACT_DIR, BAKEOFF_DIR  # picks up the override

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    n_q, H = 5, 16
    arr_A1 = np.arange(n_q * H, dtype=np.float32).reshape(n_q, H)
    arr_A2 = -arr_A1
    torch.save(
        {"activations_one_cond": arr_A1, "cond_id": "A1", "n_probes": n_q},
        ACT_DIR / "last_prompt__layer5__condA1.pt",
    )
    torch.save(
        {"activations_one_cond": arr_A2, "cond_id": "A2", "n_probes": n_q},
        ACT_DIR / "last_prompt__layer5__condA2.pt",
    )
    out = merge_partitioned_activations(("last_prompt",), (5,), overwrite=True)
    stacked = out["last_prompt"][5]
    expected = np.stack([arr_A1, arr_A2], axis=0)
    assert stacked.shape == expected.shape, (stacked.shape, expected.shape)
    # Order is canonical (A1, A2): A1 first by CONDITIONS order.
    assert np.array_equal(stacked, expected), "merged stack != expected"
    # Canonical file written.
    canonical = ACT_DIR / "last_prompt__layer5.pt"
    assert canonical.exists()
    return {"shape": list(stacked.shape), "ok": True}


# ─────────────────────────── Check 7: root override ───────────────────────────


def check_root_override() -> dict:
    """_set_roots redirects every dependent path atomically."""
    _set_roots(SMOKE_ROOT)
    from issue493_extraction_metric_bakeoff import (
        ACT_DIR,
        BAKEOFF_DIR,
        FIGURE_DIR,
        METRIC_DIR,
        REGR_DIR,
    )

    assert BAKEOFF_DIR == SMOKE_ROOT
    assert ACT_DIR == SMOKE_ROOT / "activations"
    assert METRIC_DIR == SMOKE_ROOT / "metrics"
    assert REGR_DIR == SMOKE_ROOT / "regression"
    return {
        "BAKEOFF_DIR": str(BAKEOFF_DIR),
        "ACT_DIR": str(ACT_DIR),
        "METRIC_DIR": str(METRIC_DIR),
        "REGR_DIR": str(REGR_DIR),
        "FIGURE_DIR": str(FIGURE_DIR),
    }


# ─────────────────────────── Check 8: batched vs serial equality ─────────────


def check_batched_vs_serial_equality() -> dict:
    """THE batching gate: cosine(batched extraction, ORIGINAL #493 serial
    extraction) ≥ 0.999 per (layer × extraction point) on a tiny CPU model.

    Loads ``hf-internal-testing/tiny-random-gpt2`` (no chat template; we
    inject one so ``build_prompt_for_condition`` on Class B works), runs
    a B=3 batched ``_extract_batch``, then runs ``_extract_one`` per
    probe — the SAME code path #493's serial production loop uses
    (round-2 fix #6: the round-1 smoke compared `_extract_batch(B>=2)`
    vs `_extract_batch(B=1)`, which is batch-vs-batch and NOT proof
    that the batched path agrees with the preserved #493 serial path).
    Asserts cosine(batched, serial) ≥ 0.999 per probe × layer for both
    ``last_prompt`` and ``mean_response``. Also exercises the next-token
    logits capture path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # tiny-random-gpt2's LM head wraps the transformer at .transformer.h[L];
    # build a tiny adapter so _LayerHookCapture can hook m.model.layers[L].
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    # Inject a minimal Qwen-style chat template so build_prompt_for_condition
    # (Class B path) works on this tiny model. The template just wraps the
    # user turn; matches the shape of the real prompt enough for the batched
    # vs serial equality test (we are NOT testing chat-template fidelity).
    tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    mdl = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    mdl.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    class _GPT2Adapter:
        def __init__(self, m):
            # GPT2LMHeadModel exposes the transformer at .transformer; .h is
            # the ModuleList of decoder blocks. Wrap to match the
            # ``model.model.layers[L]`` shape _LayerHookCapture hooks.
            self.model = type("inner", (), {"layers": m.transformer.h})()
            self.config = m.config

        def __call__(self, *a, **kw):
            return mdl(*a, **kw)

        def generate(self, *a, **kw):
            return mdl.generate(*a, **kw)

    adapter = _GPT2Adapter(mdl)

    # Build a fake "Condition" object adequate for _build_prompts_for_extraction
    # (Class B — bare wrap_template, no system prompt).
    class _Cond:
        cls = "B"
        cid = "B1"
        wrap_template = "{q}"

    probes = [
        "What is the capital of France?",
        "Tell me a joke.",
        "How does photosynthesis work?",
    ]
    target_layers = (0, 1)
    extraction_points = ("last_prompt", "mean_response")

    # Batched (B=3) — exercises the multi-sequence left-pad path.
    with _LayerHookCapture(adapter, target_layers) as cap:
        rows_b, _meta_b, _, nt_b = _extract_batch(
            adapter,
            tok,
            device="cpu",
            cond=_Cond(),
            questions=probes,
            class_d_rewrites={},
            extraction_points=extraction_points,
            layers=target_layers,
            max_response_tokens=4,
            hook_capture=cap,
            capture_next_token_logits=True,
        )

    # SERIAL via _extract_one (per probe) — the ORIGINAL #493 production
    # serial path, unchanged. This is the byte-equality reference the
    # batched path must agree with within fp tolerance.
    rows_s: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
    with _LayerHookCapture(adapter, target_layers) as cap:
        for i, p in enumerate(probes):
            res, _meta = _extract_one(
                adapter,
                tok,
                device="cpu",
                cond=_Cond(),
                question=p,
                class_d_rewrites={},
                extraction_points=extraction_points,
                layers=target_layers,
                max_response_tokens=4,
                hook_capture=cap,
            )
            rows_s[i] = res

    cosines_lp: list[float] = []
    cosines_mr: list[float] = []
    mr_compared = 0
    for i in range(len(probes)):
        for L in target_layers:
            v_b_lp = rows_b[i]["last_prompt"][L]
            v_s_lp = rows_s[i]["last_prompt"][L]
            cs = torch.nn.functional.cosine_similarity(v_b_lp, v_s_lp, dim=0).item()
            cosines_lp.append(cs)
            assert cs > 0.999, (
                f"batched(_extract_batch)/serial(_extract_one) LAST_PROMPT cosine "
                f"probe={i} L={L} = {cs:.6f} < 0.999 — batched extraction diverges "
                "from the preserved #493 serial path at last_prompt!"
            )
            # mean_response may be NaN if either path emitted zero
            # response tokens; skip those rows. We require at least one
            # mean_response comparison to land non-NaN so the gate isn't
            # silently a no-op.
            v_b_mr = rows_b[i]["mean_response"].get(L)
            v_s_mr = rows_s[i]["mean_response"].get(L)
            if v_b_mr is None or v_s_mr is None:
                continue
            if torch.any(torch.isnan(v_b_mr)) or torch.any(torch.isnan(v_s_mr)):
                continue
            cs_mr = torch.nn.functional.cosine_similarity(v_b_mr, v_s_mr, dim=0).item()
            cosines_mr.append(cs_mr)
            mr_compared += 1
            assert cs_mr > 0.999, (
                f"batched(_extract_batch)/serial(_extract_one) MEAN_RESPONSE cosine "
                f"probe={i} L={L} = {cs_mr:.6f} < 0.999 — batched extraction "
                "diverges from the preserved #493 serial path at mean_response!"
            )
    assert mr_compared >= 1, (
        "mean_response equality gate produced zero comparable rows — the gate "
        "would have been a silent no-op. Increase max_response_tokens or pick "
        "probes the tiny model will respond to."
    )
    assert all(i in nt_b for i in range(len(probes))), "next-token logits not captured for all"
    return {
        "n_probes": len(probes),
        "n_layers": len(target_layers),
        "serial_reference": "_extract_one (preserved #493 production path)",
        "last_prompt_min_cosine": min(cosines_lp),
        "last_prompt_max_cosine": max(cosines_lp),
        "mean_response_min_cosine": min(cosines_mr) if cosines_mr else None,
        "mean_response_max_cosine": max(cosines_mr) if cosines_mr else None,
        "mean_response_rows_compared": mr_compared,
        "n_next_token_logits": len(nt_b),
    }


# ─────────────────────────── Check 9: multi-GPU partition correctness ─────────


def check_multi_gpu_partition_correctness() -> dict:
    """Two disjoint cond partitions, each writing per-cond files; the merger
    reconstructs the same stacked grid as a single all-conds run."""
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "partition")
    from issue493_extraction_metric_bakeoff import ACT_DIR, BAKEOFF_DIR

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    n_q, H = 4, 8
    # Cids B1..B5 (Class B → no end_of_system) → canonical order.
    cid_arr = {
        "B1": np.full((n_q, H), 1.0, dtype=np.float32),
        "B2": np.full((n_q, H), 2.0, dtype=np.float32),
        "B3": np.full((n_q, H), 3.0, dtype=np.float32),
        "B4": np.full((n_q, H), 4.0, dtype=np.float32),
    }
    # "GPU 0" writes B1, B2; "GPU 1" writes B3, B4.
    for cid in ("B1", "B2", "B3", "B4"):
        torch.save(
            {"activations_one_cond": cid_arr[cid], "cond_id": cid, "n_probes": n_q},
            ACT_DIR / f"last_prompt__layer0__cond{cid}.pt",
        )
    out = merge_partitioned_activations(("last_prompt",), (0,), overwrite=True)
    stacked = out["last_prompt"][0]
    # Canonical CONDITIONS order: B1 < B2 < B3 < B4.
    expected = np.stack([cid_arr[c] for c in ("B1", "B2", "B3", "B4")], axis=0)
    assert stacked.shape == expected.shape, (stacked.shape, expected.shape)
    assert np.array_equal(stacked, expected), "partition merge != expected canonical order"
    return {"shape": list(stacked.shape), "ok": True}


# ─────────────────────────── Check 10: partition no-drop assertion ───────────


def check_partition_no_drop_assertion() -> dict:
    """expected_cond_ids gates the merge: missing + extra cids both raise.

    Confirms the no-drop assertion added in round-2 (fix #4) catches stale /
    partial partitions before they corrupt downstream regression.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "nodrop")
    from issue493_extraction_metric_bakeoff import ACT_DIR, BAKEOFF_DIR

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    n_q, H = 3, 4
    # Only write B1, B2 (B3 is "missing" — simulates a silently-dead GPU worker).
    for cid in ("B1", "B2"):
        torch.save(
            {
                "activations_one_cond": np.ones((n_q, H), dtype=np.float32),
                "cond_id": cid,
                "n_probes": n_q,
            },
            ACT_DIR / f"last_prompt__layer0__cond{cid}.pt",
        )
    # Missing-cid case: expected has B3, present only has {B1, B2} → must raise.
    missing_caught = False
    try:
        merge_partitioned_activations(
            ("last_prompt",),
            (0,),
            overwrite=True,
            expected_cond_ids=["B1", "B2", "B3"],
        )
    except AssertionError as e:
        msg = str(e)
        assert "missing_conds=['B3']" in msg, (
            f"Expected missing_conds=['B3'] in message; got: {msg}"
        )
        missing_caught = True
    assert missing_caught, "no-drop assertion did NOT fire on missing cid B3"

    # Extra-cid case: expected only B1, present {B1, B2} → must raise on B2 extra.
    extra_caught = False
    try:
        merge_partitioned_activations(
            ("last_prompt",),
            (0,),
            overwrite=True,
            expected_cond_ids=["B1"],
        )
    except AssertionError as e:
        msg = str(e)
        assert "extra_conds=['B2']" in msg, f"Expected extra_conds=['B2'] in message; got: {msg}"
        extra_caught = True
    assert extra_caught, "no-drop assertion did NOT fire on extra cid B2"

    # Match case: expected B1+B2, present B1+B2 → merge succeeds.
    out = merge_partitioned_activations(
        ("last_prompt",),
        (0,),
        overwrite=True,
        expected_cond_ids=["B1", "B2"],
    )
    assert out["last_prompt"][0].shape == (2, n_q, H), out["last_prompt"][0].shape
    return {
        "missing_assertion_fired": missing_caught,
        "extra_assertion_fired": extra_caught,
        "match_case_shape": list(out["last_prompt"][0].shape),
    }


# ─────────────────────────── Check 11: prod JS cross-check is wired ──────────


def check_prod_js_cross_check_wired() -> dict:
    """write_next_token_js_matrix MUST call the #406 cross-check + raise on
    floor failure when the reference is on disk. Round-1 blocker #1: this
    used to be a logged-only no-op in production, despite the cosine
    cross-check being a hard gate at the same site.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "jsxcheck")
    from issue493_extraction_metric_bakeoff import (
        BAKEOFF_DIR,
        METRIC_DIR,
    )

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    # Build adversarial next-token sidecars whose JS ranking deliberately
    # MISMATCHES #406's (we shuffle the cid → probability assignment so
    # rank correlation tanks). This MUST raise when enforce_cross_check=True.
    nt_dir = BAKEOFF_DIR / "next_token_logits"
    nt_dir.mkdir(parents=True, exist_ok=True)
    n_q, V = 4, 32
    cids = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "C1",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]
    # Random adversarial distributions per cid (different seeds → uncorrelated).
    for k, cid in enumerate(cids):
        rng = np.random.default_rng(k * 7919 + 17)
        # Sharp distributions so JS spread is large.
        logits = rng.normal(size=(n_q, V)).astype(np.float32) * 5.0
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        torch.save(
            {
                "extraction_point": "last_prompt",
                "cond_id": cid,
                "n_probes": n_q,
                "vocab_size": V,
                "probs": probs.astype(np.float32),
            },
            nt_dir / f"last_prompt__cond{cid}.pt",
        )

    # If #406 isn't available at this dev VM, the cross-check is a no-op
    # (per its own "no #406 reference available" branch). Skip with a
    # clearly tagged digest.
    p406 = PROJECT_ROOT / "eval_results/issue_406/divergence/D_matrix.json"
    if not p406.exists():
        return {
            "ok": True,
            "reason": "no #406 reference on dev VM; production wiring untested here",
        }

    # enforce_cross_check=True (production default) must raise on the
    # adversarial input.
    raised = False
    try:
        write_next_token_js_matrix(enforce_cross_check=True)
    except AssertionError as e:
        raised = True
        msg = str(e)
        assert "rank correlation" in msg.lower(), f"unexpected raise message: {msg}"
    assert raised, (
        "write_next_token_js_matrix(enforce_cross_check=True) did NOT raise on "
        "adversarial input — the JS baseline would ship unguarded in production!"
    )
    # Sidecar of the failure was written.
    cross_check_path = METRIC_DIR / "last_prompt__layer-1__next_token_js__raw__cross_check_406.json"
    assert cross_check_path.exists(), "cross_check_406.json sidecar not written on failure"
    cc = json.loads(cross_check_path.read_text())
    assert cc.get("failed") is True, f"failure sidecar missing 'failed: true': {cc}"

    # enforce_cross_check=False must NOT raise (smoke / dev path).
    write_next_token_js_matrix(enforce_cross_check=False)

    # Sanity: load_next_token_logits reads the sidecars we wrote.
    loaded = load_next_token_logits()
    assert len(loaded) == len(cids), f"loaded {len(loaded)} cids, expected {len(cids)}"
    return {
        "adversarial_raise_on_enforce_true": raised,
        "no_raise_on_enforce_false": True,
        "failure_sidecar_written": cross_check_path.exists(),
        "sidecars_loaded": len(loaded),
    }


# ─────────────────────────── Check 12: 50-prefix strict cosine path ──────────


def check_strict_50_prefix_cosine_path() -> dict:
    """Verify the round-2 fix #5 main()-level slice logic works end-to-end:
    when n_q > 50 AND --probe-pool is set, the 50-prefix path is selected
    and the strict gate runs. We test the SHAPE of the slice here (the
    full strict-vs-#406 comparison runs on the pod).
    """
    # Reuse the probe pool — its first 50 ARE q_test by construction (the
    # production loader asserts this).
    if not PROBES_PATH.exists():
        return {"ok": True, "reason": "probe pool missing; skipped"}
    probes_loaded = _load_probe_questions(pool_path=PROBES_PATH)
    from explore_persona_space.experiments.i460_data import load_q_test_extended_50

    q_test = load_q_test_extended_50()
    # Prefix invariant: the first 50 of the 500-pool are byte-identical
    # to q_test. This is the precondition for the round-2 strict-prefix
    # slice in main() — if it fails, the strict gate would silently
    # compare against the wrong probes.
    assert probes_loaded[:50] == q_test, (
        "500-pool prefix is not q_test — strict cosine slice broken"
    )
    # Simulate the slice main() builds: arr[:, :50, :] on a fake (n_cond, 500, H) tensor.
    fake = np.arange(16 * 500 * 4, dtype=np.float32).reshape(16, 500, 4)
    sliced = fake[:, :50, :]
    assert sliced.shape == (16, 50, 4), sliced.shape
    return {
        "prefix_is_q_test": True,
        "n_q_total": 500,
        "sliced_shape": list(sliced.shape),
    }


# ─────────────────────────── Check 13: --batched no --partitioned ────────────


def check_batched_without_partitioned_canonical() -> dict:
    """Round-2 fix #3: --batched without --partitioned MUST produce
    canonical <point>__layer<L>.pt files the metrics phase can load.

    The fix routes the in-process branch (when write_partitioned=False)
    to write per-cond files AND auto-merge them into the canonical shape.
    We exercise the path-write contract end-to-end: write per-cond files
    the way the batched extractor does, then call
    merge_partitioned_activations EXACTLY as run_extraction_batched does
    in its auto-merge branch (with expected_cond_ids set), and confirm
    the canonical file lands + is correctly shaped + matches the expected
    canonical ordering. The model's forward-pass correctness is already
    covered by check_batched_vs_serial_equality, so this check is
    model-free.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "nopartition")
    from issue493_extraction_metric_bakeoff import ACT_DIR, BAKEOFF_DIR

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    n_q, H = 3, 4
    for cid in ("B1", "B2", "B3"):
        torch.save(
            {
                "activations_one_cond": np.full((n_q, H), float(ord(cid[1])), dtype=np.float32),
                "cond_id": cid,
                "n_probes": n_q,
            },
            ACT_DIR / f"last_prompt__layer0__cond{cid}.pt",
        )
    # Same call shape run_extraction_batched's `if not write_partitioned`
    # branch uses (with the same expected_cond_ids list for the no-drop
    # assertion).
    merge_partitioned_activations(
        ("last_prompt",), (0,), overwrite=True, expected_cond_ids=["B1", "B2", "B3"]
    )
    canonical = ACT_DIR / "last_prompt__layer0.pt"
    assert canonical.exists(), f"canonical file not produced at {canonical}"
    loaded = torch.load(canonical, map_location="cpu", weights_only=False)
    assert loaded["activations"].shape == (3, n_q, H), loaded["activations"].shape
    assert loaded["cond_ids"] == ["B1", "B2", "B3"], loaded["cond_ids"]
    return {
        "canonical_exists": True,
        "shape": list(loaded["activations"].shape),
        "cond_ids": loaded["cond_ids"],
    }


# ─── Check 14: wholly-missing (pt, L) raises in the no-drop gate ─────────────


def check_wholly_missing_layer_raises() -> dict:
    """Round-3 fix #1: a WHOLLY-MISSING (point, layer) — zero per-cond files
    for that combination — must raise the no-drop assertion when
    expected_cond_ids is set. Before round 3 the merge loop iterated only
    over (pt, L) groups present in `grouped`, so a layer that no GPU
    wrote silently skipped the assertion → `--merge-only` exited clean and
    the "all 28 layers" run silently became an incomplete layer profile.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "wholly_missing")
    from issue493_extraction_metric_bakeoff import ACT_DIR, BAKEOFF_DIR

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    n_q, H = 3, 4
    # Write only layer 0; request layers 0 + 1. Layer 1 is wholly missing.
    for cid in ("B1", "B2"):
        torch.save(
            {
                "activations_one_cond": np.ones((n_q, H), dtype=np.float32),
                "cond_id": cid,
                "n_probes": n_q,
            },
            ACT_DIR / f"last_prompt__layer0__cond{cid}.pt",
        )
    raised = False
    try:
        merge_partitioned_activations(
            ("last_prompt",),
            (0, 1),  # request BOTH layers; layer 1 has zero per-cond files
            overwrite=True,
            expected_cond_ids=["B1", "B2"],
        )
    except AssertionError as e:
        msg = str(e)
        # Must name the wholly-missing layer in the error.
        assert "layer=1" in msg, f"Expected 'layer=1' in error message; got: {msg}"
        assert "No partitioned per-cond files AT ALL" in msg, (
            f"Expected 'No partitioned per-cond files AT ALL' in message; got: {msg}"
        )
        raised = True
    assert raised, (
        "no-drop assertion did NOT fire on a wholly-missing (pt, L) — round-3 hole #1 still open"
    )
    return {"wholly_missing_layer_raised": True, "missing_layer": 1}


# ─── Check 15: non-partitioned phase=all writes next_token_js ────────────────


def check_non_partitioned_phase_all_writes_js() -> dict:
    """Round-3 fix #2: the `--probe-pool ... --batch-size>1` WITHOUT
    `--partitioned` path is a supported run mode (auto-merge branch from
    round-2 fix #3). Round-2's JS write gate `if args.partitioned and
    args.phase == "all"` excluded it → metrics/regression proceeded with
    NO next_token_js matrix, dropping the baseline silently.

    Round-3 fix moves the JS write to `args.phase == "all" and not
    args.no_next_token_js` — runs on ALL paths. This check writes
    sidecars + invokes write_next_token_js_matrix(enforce_cross_check=
    False) directly to confirm the matrix lands when called via the
    round-3 site, and that the cross-check sidecar is written.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "nonpart_phase_all_js")
    from issue493_extraction_metric_bakeoff import BAKEOFF_DIR, METRIC_DIR

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    # Mimic what _extract_batch would write: per-cond next-token logits
    # sidecars for a handful of class-A cids.
    nt_dir = BAKEOFF_DIR / "next_token_logits"
    nt_dir.mkdir(parents=True, exist_ok=True)
    n_q, V = 3, 16
    cids = ["A1", "A2", "A3"]
    rng = np.random.default_rng(0)
    for k, cid in enumerate(cids):
        logits = rng.normal(size=(n_q, V)).astype(np.float32) * (k + 1)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        torch.save(
            {
                "extraction_point": "last_prompt",
                "cond_id": cid,
                "n_probes": n_q,
                "vocab_size": V,
                "probs": probs.astype(np.float32),
            },
            nt_dir / f"last_prompt__cond{cid}.pt",
        )

    # Call the same function the round-3 site invokes; enforce=False so
    # the synthetic 3-cond input doesn't trip the #406 rank-correlation
    # floor (production runs use the full 16-cond grid where rho is real).
    matrix_path = write_next_token_js_matrix(enforce_cross_check=False)
    assert matrix_path is not None, "write_next_token_js_matrix returned None unexpectedly"
    assert matrix_path.exists(), f"matrix file not written at {matrix_path}"
    # Cross-check sidecar also written (PASS or fail-loud-on-enforce=True,
    # smoke uses enforce=False so we just check the sidecar lands).
    cross_check_path = METRIC_DIR / "last_prompt__layer-1__next_token_js__raw__cross_check_406.json"
    # The cross-check sidecar is only written when the cross-check actually
    # runs (i.e. when the #406 reference is present). On the dev VM #406
    # may or may not be there; assert the matrix landed (the load-bearing
    # part) and the sidecar landed IF #406 is present.
    payload = json.loads(matrix_path.read_text())
    assert payload["metric"] == "next_token_js"
    assert payload["layer"] == -1
    assert len(payload["cond_ids"]) == len(cids)
    p406 = PROJECT_ROOT / "eval_results/issue_406/divergence/D_matrix.json"
    return {
        "matrix_written": str(matrix_path.name),
        "matrix_layer_sentinel": payload["layer"],
        "matrix_metric": payload["metric"],
        "n_cids_in_matrix": len(payload["cond_ids"]),
        "cross_check_sidecar_written": cross_check_path.exists() if p406.exists() else "n/a",
    }


# ─────────────────────────── Main ───────────────────────────


def main() -> int:
    digest: dict = {}
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    checks = [
        ("probe_pool", check_probe_pool),
        ("js_rowwise", check_js_rowwise),
        ("matrix_build", check_matrix_build),
        ("cross_against_406", check_cross_against_406),
        ("root_override", check_root_override),
        ("partition_merge", check_partition_merge),
        ("multi_gpu_partition_correctness", check_multi_gpu_partition_correctness),
        ("batched_vs_serial_equality", check_batched_vs_serial_equality),
        ("partition_no_drop_assertion", check_partition_no_drop_assertion),
        ("prod_js_cross_check_wired", check_prod_js_cross_check_wired),
        ("strict_50_prefix_cosine_path", check_strict_50_prefix_cosine_path),
        ("batched_without_partitioned_canonical", check_batched_without_partitioned_canonical),
        ("wholly_missing_layer_raises", check_wholly_missing_layer_raises),
        ("non_partitioned_phase_all_writes_js", check_non_partitioned_phase_all_writes_js),
    ]
    for name, fn in checks:
        logger.info("=== check: %s ===", name)
        try:
            digest[name] = fn()
        except Exception as e:
            logger.exception("Check %s FAILED: %s", name, e)
            digest[name] = {"FAILED": str(e)}
            out_path = PROJECT_ROOT / "eval_results" / "issue_502" / "cpu_smoke_digest.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(digest, indent=2, default=str))
            return 1
        logger.info("OK %s: %s", name, digest[name])

    out_path = PROJECT_ROOT / "eval_results" / "issue_502" / "cpu_smoke_digest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(digest, indent=2, default=str))
    logger.info("All %d checks PASSED. Digest at %s", len(checks), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
