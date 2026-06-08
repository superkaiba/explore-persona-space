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


def check_non_partitioned_phase_all_writes_js() -> dict:  # noqa: C901 — AST inspection branches + helper invocation + per-call gate validation; flattening would just inline the inspection helpers.
    """Round-3 fix #2 — REWRITTEN in round-4 to test the WIRING, not the
    helper.

    Round-3 fix moved the JS write to `args.phase == "all" and not
    args.no_next_token_js` so the non-partitioned auto-merge path also
    writes the JS matrix. The round-3 version of this smoke called
    write_next_token_js_matrix() directly, which is tautological — it
    would have passed even on the round-2-broken code where main() never
    invoked the helper for the non-partitioned phase=all path. (Codex
    round-3 major #2.)

    Round-4 round 4 version: STATICALLY inspect main()'s source for the
    canonical-wiring shape — the JS write call must live under a gate
    of the form `args.phase == "all" and not args.no_next_token_js`
    (or equivalent), NOT under `args.partitioned and ...`. We also
    confirm the helper still produces the matrix file when called the
    way main() calls it (so the helper isn't separately broken).
    Combination = wiring + behavior, both required for the path to fire.
    """
    import ast
    import inspect
    import shutil

    import issue493_extraction_metric_bakeoff as bakeoff
    import torch

    # ── WIRING TEST: AST inspection of main() ──
    src = inspect.getsource(bakeoff.main)
    tree = ast.parse(src)

    def _is_phase_all_gate(node: ast.AST) -> bool:
        """Detect an `args.phase == "all"` clause (LHS or RHS), with or
        without an `and not args.no_next_token_js` sibling — anything
        gated on `args.partitioned` for the JS call is the round-2 bug."""
        return (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and (
                (
                    isinstance(node.left, ast.Attribute)
                    and node.left.attr == "phase"
                    and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value == "all"
                )
                or (
                    isinstance(node.comparators[0], ast.Attribute)
                    and node.comparators[0].attr == "phase"
                    and isinstance(node.left, ast.Constant)
                    and node.left.value == "all"
                )
            )
        )

    def _has_attr_in_subtree(node: ast.AST, attr: str) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == attr:
                return True
        return False

    # Build a parent map so we can find the INNERMOST enclosing `If` for
    # each call site (walking-from-the-root only finds the outermost if).
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent

    def _all_enclosing_ifs(call_node: ast.Call) -> list[ast.If]:
        """Walk UP from the call to collect every enclosing `If` (innermost
        first). The call is gated by the conjunction of every test."""
        out: list[ast.If] = []
        cur: ast.AST | None = parent_map.get(id(call_node))
        while cur is not None:
            if isinstance(cur, ast.If):
                out.append(cur)
            cur = parent_map.get(id(cur))
        return out

    js_call_gates: list[str] = []
    js_call_count = 0
    has_merge_only_site = False
    has_phase_all_site = False
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_next_token_js_matrix"
        ):
            continue
        js_call_count += 1
        enclosing = _all_enclosing_ifs(node)
        if not enclosing:
            js_call_gates.append("<top-level — no enclosing if>")
            continue
        # Conjunction of every enclosing `if test:` — the call fires only
        # when EVERY test is True. The most-specific (innermost) test is
        # also the one most likely to carry the phase-all / merge_only
        # routing, but we walk all of them so a refactor that splits the
        # gate across nested ifs still passes.
        gate_srcs = [ast.unparse(eif.test) for eif in enclosing]
        is_phase_all = any(_is_phase_all_gate(t) for eif in enclosing for t in ast.walk(eif.test))
        is_merge_only = any(_has_attr_in_subtree(eif.test, "merge_only") for eif in enclosing)
        has_partitioned = any(_has_attr_in_subtree(eif.test, "partitioned") for eif in enclosing)
        js_call_gates.append(" and ".join(f"({s})" for s in gate_srcs))
        if is_phase_all:
            has_phase_all_site = True
        if is_merge_only:
            has_merge_only_site = True
        # Reject the round-2 bug shape: a call gated on `args.partitioned`
        # WITHOUT phase=="all" AND WITHOUT merge_only.
        if has_partitioned and not is_phase_all and not is_merge_only:
            raise AssertionError(
                f"write_next_token_js_matrix call is gated on `args.partitioned` "
                f'WITHOUT the phase=="all" or merge_only gate — this is the '
                f"round-2 bug shape the round-3 fix was supposed to remove. "
                f"Gates: {gate_srcs!r}"
            )
        # Reject any other unexpected gate shape.
        if not (is_phase_all or is_merge_only):
            raise AssertionError(
                f"write_next_token_js_matrix call has an unexpected enclosing "
                f'gate (not `args.merge_only`, not `args.phase == "all"`). '
                f"Gates: {gate_srcs!r}"
            )
    assert js_call_count >= 2, (
        f"expected ≥2 write_next_token_js_matrix call sites in main() "
        f"(--merge-only path + post-extraction phase=='all'); found {js_call_count}"
    )
    assert has_merge_only_site, (
        "no write_next_token_js_matrix call site is gated by `args.merge_only` "
        "— the --merge-only aggregation path would silently drop the JS baseline"
    )
    assert has_phase_all_site, (
        'no write_next_token_js_matrix call site is gated by `args.phase == "all"` '
        "— the round-3 fix wiring is missing; the non-partitioned auto-merge path "
        "would silently drop the JS baseline"
    )

    # ── BEHAVIOR TEST: the helper still produces the matrix when called ──
    # the way main() calls it (so a regression in the helper itself is
    # also caught here).
    _set_roots(SMOKE_ROOT / "nonpart_phase_all_js")
    from issue493_extraction_metric_bakeoff import BAKEOFF_DIR

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
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
    matrix_path = write_next_token_js_matrix(enforce_cross_check=False)
    assert matrix_path is not None and matrix_path.exists()
    payload = json.loads(matrix_path.read_text())
    assert payload["metric"] == "next_token_js"
    assert payload["layer"] == -1
    return {
        "n_js_call_sites_in_main": js_call_count,
        "js_call_gates": js_call_gates,
        "matrix_metric": payload["metric"],
        "matrix_layer_sentinel": payload["layer"],
        "wiring_test_passed": True,
        "behavior_test_passed": True,
    }


def check_prod_js_cross_check_sidecar_asserted() -> dict:
    """Round-4 fix #3 (Codex round-3 major #3): the round-3 smoke
    `check_prod_js_cross_check_wired` previously RECORDED whether the
    sidecar was written when #406 exists (`"cross_check_sidecar_written":
    cross_check_path.exists() if p406.exists() else "n/a"`) instead of
    ASSERTING it. A regression that stopped writing the sidecar would
    have shown up as `False` in the digest but the smoke would still
    PASS. This new check ASSERTS the sidecar lands (with the floor
    metadata) when #406 is on disk, so a missing sidecar fails the
    smoke loudly.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "js_sidecar_asserted")
    from issue493_extraction_metric_bakeoff import BAKEOFF_DIR, METRIC_DIR

    p406 = PROJECT_ROOT / "eval_results/issue_406/divergence/D_matrix.json"
    if not p406.exists():
        # No reference → cross-check is a no-op by design.
        # Nothing to assert in that environment; skip with a clear tag.
        return {
            "ok": True,
            "reason": "no #406 reference on dev VM; sidecar assertion skipped",
        }

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    # Build a benign 16-cond sidecar set so the rank-correlation comes
    # out high (above the floor). Each cid's probs are a translation of
    # the SAME monotone profile, with cid-indexed mean offset — produces
    # a clean monotone JS structure that correlates with #406's JS.
    nt_dir = BAKEOFF_DIR / "next_token_logits"
    nt_dir.mkdir(parents=True, exist_ok=True)
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
    n_q, V = 4, 32
    base_logits = np.linspace(-2.0, 2.0, V, dtype=np.float32)
    for k, cid in enumerate(cids):
        # cid-indexed monotone shift so JS scales linearly with |i - j|.
        shifted = base_logits + (k - len(cids) / 2.0) * 0.5
        rep = np.tile(shifted[None, :], (n_q, 1))
        probs = np.exp(rep - rep.max(axis=1, keepdims=True))
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

    matrix_path = write_next_token_js_matrix(enforce_cross_check=False)
    assert matrix_path is not None and matrix_path.exists()
    # ASSERT (not record) the sidecar exists when #406 is on disk.
    cross_check_path = METRIC_DIR / "last_prompt__layer-1__next_token_js__raw__cross_check_406.json"
    assert cross_check_path.exists(), (
        f"#406 cross-check sidecar NOT written at {cross_check_path} despite "
        f"the #406 reference being present at {p406}. A regression that "
        "stopped writing the sidecar would silently bypass the JS safety net."
    )
    cc = json.loads(cross_check_path.read_text())
    # The sidecar must carry the rank-corr floor metadata so a reviewer
    # can audit it.
    if cc.get("failed") is False:
        summary = cc.get("summary", {})
        assert "rank_corr_floor" in summary, (
            f"cross-check sidecar summary missing 'rank_corr_floor': {summary}"
        )
        assert "rank_corr_spearman" in summary, (
            f"cross-check sidecar summary missing 'rank_corr_spearman': {summary}"
        )
        floor_val = summary["rank_corr_floor"]
        rho_val = summary["rank_corr_spearman"]
    else:
        # On a failed-but-not-raised path (enforce=False), the sidecar
        # records the failure reason + floor at the top level.
        assert "rank_corr_floor" in cc, f"failed sidecar missing 'rank_corr_floor': {cc}"
        floor_val = cc["rank_corr_floor"]
        rho_val = None
    return {
        "p406_reference_present": True,
        "sidecar_exists": True,
        "sidecar_failed": cc.get("failed"),
        "rank_corr_floor": floor_val,
        "rank_corr_spearman": rho_val,
    }


# ─── Check 16: cache-bypass holistic-gate scenarios (Round-4 fix #1) ─────────


def check_cache_bypass_stale_canonical_raises() -> dict:
    """Round-4 fix #1 (Critical): cache-hit short-circuit must validate
    against expected_cond_ids BEFORE accepting the cached canonical.

    Scenario A: a stale canonical (cond_ids = {B1, B2, B3}) exists from a
    prior run; current run has ZERO per-cond files for that (pt, L); no
    --overwrite. With expected=["B1", "B2", "B3"], the cache should match
    and reuse cleanly. With expected=["B1", "B2", "B3", "B4"], the cache
    is stale (missing B4) and must RAISE — round 3 silently reused it.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "cache_bypass_stale")
    from issue493_extraction_metric_bakeoff import ACT_DIR, BAKEOFF_DIR

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    # Plant a stale canonical with cond_ids={B1, B2, B3}, NO per-cond files.
    n_q, H = 3, 4
    stale_arr = np.ones((3, n_q, H), dtype=np.float32)
    torch.save(
        {
            "schema_version": 1,
            "extraction_point": "last_prompt",
            "layer": 0,
            "cond_ids": ["B1", "B2", "B3"],
            "n_probes": n_q,
            "hidden_size": H,
            "activations": stale_arr,
        },
        ACT_DIR / "last_prompt__layer0.pt",
    )

    # Scenario A: expected matches the stale canonical → cache reuses cleanly.
    out_match = merge_partitioned_activations(
        ("last_prompt",),
        (0,),
        overwrite=False,
        expected_cond_ids=["B1", "B2", "B3"],
    )
    assert "last_prompt" in out_match and 0 in out_match["last_prompt"]
    assert out_match["last_prompt"][0].shape == (3, n_q, H)

    # Scenario B: expected has an extra cid B4 NOT in the stale canonical
    # AND no per-cond file for B4. Round-3 reused the cache silently; round-4
    # must RAISE.
    raised_b = False
    try:
        merge_partitioned_activations(
            ("last_prompt",),
            (0,),
            overwrite=False,
            expected_cond_ids=["B1", "B2", "B3", "B4"],
        )
    except AssertionError as e:
        msg = str(e)
        assert "Cached canonical" in msg, f"unexpected raise message: {msg}"
        assert "missing_conds=['B4']" in msg, f"expected missing_conds=['B4'] in: {msg}"
        raised_b = True
    assert raised_b, (
        "ROUND-4 fix #1 still open: cache-hit short-circuit silently reused a "
        "stale canonical missing B4 instead of raising"
    )

    # Scenario C: same stale canonical, but expected has a DIFFERENT set
    # (missing one + adding another) → must raise.
    raised_c = False
    try:
        merge_partitioned_activations(
            ("last_prompt",),
            (0,),
            overwrite=False,
            expected_cond_ids=["B1", "B2", "B5"],
        )
    except AssertionError as e:
        msg = str(e)
        assert "missing_conds=['B5']" in msg and "extra_conds=['B3']" in msg, (
            f"expected missing=['B5'] + extra=['B3'] in: {msg}"
        )
        raised_c = True
    assert raised_c, "expected raise on cache-vs-expected set disagreement"

    return {
        "scenario_a_match_reuses_cache": True,
        "scenario_b_stale_missing_b4_raised": True,
        "scenario_c_set_mismatch_raised": True,
    }


# ─── Check 17: holistic validator catches a stale partial canonical ─────────


def check_validate_canonical_completeness_gate() -> dict:
    """ROUND-4 the standalone read-time validator. Plant a tree that LOOKS
    populated (canonical files exist) but is partially stale — one (pt, L)
    canonical has the right cond_ids, another has missing cids, a third is
    wholly absent. The validator must raise naming every offender so
    --phase metrics / --phase regress / --skip-extract over a partial run
    never proceeds silently.
    """
    import shutil

    import torch

    _set_roots(SMOKE_ROOT / "validator_gate")
    from issue493_extraction_metric_bakeoff import (
        ACT_DIR,
        BAKEOFF_DIR,
        validate_canonical_completeness,
    )

    if BAKEOFF_DIR.exists():
        shutil.rmtree(BAKEOFF_DIR)
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    n_q, H = 2, 4

    def _write_canonical(pt: str, L: int, cids: list[str]) -> None:
        arr = np.ones((len(cids), n_q, H), dtype=np.float32)
        torch.save(
            {
                "schema_version": 1,
                "extraction_point": pt,
                "layer": L,
                "cond_ids": cids,
                "n_probes": n_q,
                "hidden_size": H,
                "activations": arr,
            },
            ACT_DIR / f"{pt}__layer{L}.pt",
        )

    # Layer 0: good (3 cids match expected).
    _write_canonical("last_prompt", 0, ["B1", "B2", "B3"])
    # Layer 1: stale (missing B3).
    _write_canonical("last_prompt", 1, ["B1", "B2"])
    # Layer 2: wholly absent (no file).
    expected = ["B1", "B2", "B3"]

    # Match path: only layer 0 → returns cleanly.
    out_match = validate_canonical_completeness(
        ("last_prompt",),
        (0,),
        expected_cond_ids_for_point={"last_prompt": expected},
    )
    assert 0 in out_match["last_prompt"]

    # Failure path: layers (0, 1, 2) → raises naming layer 1 missing_conds
    # AND layer 2 missing file.
    raised = False
    try:
        validate_canonical_completeness(
            ("last_prompt",),
            (0, 1, 2),
            expected_cond_ids_for_point={"last_prompt": expected},
        )
    except AssertionError as e:
        msg = str(e)
        assert "layer=1" in msg, f"validator failed to name layer 1 stale: {msg}"
        assert "missing_conds=['B3']" in msg, f"layer 1 missing_conds not named: {msg}"
        assert "layer=2" in msg, f"validator failed to name layer 2 absent: {msg}"
        assert "missing canonical" in msg, f"layer 2 missing-file not named: {msg}"
        raised = True
    assert raised, "validate_canonical_completeness did NOT raise on partial / stale tree"
    return {
        "layer0_match_passes": True,
        "layer1_stale_missing_cond_raised": True,
        "layer2_missing_file_raised": True,
    }


# ─────────────── Check 18: Class-D rewrites coverage + extraction ───────────


def check_class_d_rewrites_coverage_full_probe_pool() -> dict:  # noqa: C901 — three nested integration steps; flattening would inline the inputs setup
    """ROUND-5: the runtime failure that bounced round-4 was that the
    50→500 probe-pool extension never generated Class-D rewrites for the
    450 new probes. The extraction script's Class-D code path does
    ``class_d_rewrites[question][register]`` and KeyErrors on the first
    new-probe Class-D lookup. None of the prior 17 checks exercised this
    code path on a new (post-index-49) probe.

    This check has THREE parts:
      1. **Schema check** — every question in ``probes_500.json`` appears
         in the merged Class-D dict (#406 base ∪ #502 extension), and
         every entry has all 5 registers (``formal`` / ``casual`` /
         ``indirect`` / ``declarative`` / ``enumerated``) with non-empty
         single-line rewrites.
      2. **Coverage check** — coverage extends specifically across the
         450 new probes (the failure mode was "rewrites cover only the
         first 50 q_test").
      3. **Integration check** — actually call the Class-D extraction
         code path (``_extract_batch`` with B=1) on a tiny CPU model
         against ONE new (post-index-49) probe under D1, with the
         merged rewrites dict the dispatcher's env var produces.
         This is the gate that would have caught the pod-502 failure.

    The check first regenerates the smoke rewrites extension via
    ``generate_class_d_rewrites_extension(..., smoke=True)`` if it's
    absent, so the smoke is self-contained.
    """
    import json as _json
    import os as _os

    import torch

    PROBES_PATH = PROJECT_ROOT / "eval_results" / "issue_502" / "probes_500.json"
    if not PROBES_PATH.exists():
        raise FileNotFoundError(
            f"{PROBES_PATH} missing — generate via scripts/issue502_generate_probes.py first."
        )
    probes_payload = _json.loads(PROBES_PATH.read_text())
    probes_500 = probes_payload["probes"]
    new_probes_450 = probes_payload["new_probes_450"]
    assert len(probes_500) == 500, f"probes_500.json has {len(probes_500)} probes, expected 500"
    assert len(new_probes_450) == 450, (
        f"probes_500.json has {len(new_probes_450)} new probes, expected 450"
    )

    # If the real extension exists, prefer it; otherwise materialize a
    # smoke extension on-the-fly so the check is self-contained.
    REAL_EXT = PROJECT_ROOT / "eval_results" / "issue_502" / "class_d_rewrites_extended_v1.json"
    SMOKE_EXT = (
        PROJECT_ROOT / "eval_results" / "issue_502" / "class_d_rewrites_extended_v1.smoke.json"
    )
    if REAL_EXT.exists():
        ext_path = REAL_EXT
        ext_kind = "real"
    else:
        # Lazy-import; the generator module path is sibling-scripts.
        from issue502_generate_probes import generate_class_d_rewrites_extension

        if not SMOKE_EXT.exists():
            logger.info("Materializing smoke Class-D rewrites extension at %s", SMOKE_EXT)
            generate_class_d_rewrites_extension(new_probes_450, smoke=True)
        assert SMOKE_EXT.exists(), f"smoke extension not produced at {SMOKE_EXT}"
        ext_path = SMOKE_EXT
        ext_kind = "smoke"

    # Step 1: load merged dict via the env var (the same code path the
    # extraction subprocess uses).
    from explore_persona_space.experiments.i460_data import load_class_d_rewrites

    prior_env = _os.environ.get("EPM_CLASS_D_REWRITES_EXTENSION_PATH")
    _os.environ["EPM_CLASS_D_REWRITES_EXTENSION_PATH"] = str(ext_path)
    try:
        merged = load_class_d_rewrites()
    finally:
        if prior_env is None:
            _os.environ.pop("EPM_CLASS_D_REWRITES_EXTENSION_PATH", None)
        else:
            _os.environ["EPM_CLASS_D_REWRITES_EXTENSION_PATH"] = prior_env

    # Schema check: every probe present.
    REGISTERS = ("formal", "casual", "indirect", "declarative", "enumerated")
    missing_probes = [q for q in probes_500 if q not in merged]
    if missing_probes:
        sample = missing_probes[:3]
        raise AssertionError(
            f"Class-D rewrites: {len(missing_probes)} / {len(probes_500)} probes from "
            f"probes_500.json missing from merged dict. First 3: {sample!r}"
        )

    # Schema check: all 5 registers, non-empty + single-line, for EVERY probe.
    for q in probes_500:
        by_reg = merged[q]
        for reg in REGISTERS:
            rw = by_reg.get(reg)
            if not rw or not isinstance(rw, str):
                raise AssertionError(
                    f"Class-D rewrites: probe {q!r} register {reg!r} empty / missing (got {rw!r})"
                )
            if "\n" in rw:
                raise AssertionError(
                    f"Class-D rewrites: probe {q!r} register {reg!r} multi-line: {rw!r}"
                )

    # Step 2: coverage check across the 450 new probes specifically.
    missing_new = [q for q in new_probes_450 if q not in merged]
    if missing_new:
        sample = missing_new[:3]
        raise AssertionError(
            f"Class-D rewrites: {len(missing_new)} / 450 NEW probes (post-50) "
            f"missing from merged dict. First 3: {sample!r} — this is the "
            "exact failure mode that bounced round-4 (pod-502)."
        )

    # Step 3: integration — pick one new (post-index-49) probe, run the
    # Class-D code path under D1 on a tiny CPU model with B=1.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    # Same Qwen-style chat template the batched_vs_serial check uses (Class-D
    # condition renders the rewrite as a {"role":"user","content":...} turn).
    tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    mdl.eval()

    class _GPT2Adapter:
        """Re-uses the same shape adapter as check_batched_vs_serial_equality."""

        def __init__(self, m):
            self.model = type("inner", (), {"layers": m.transformer.h})()
            self.config = m.config

        def __call__(self, *a, **kw):
            return mdl(*a, **kw)

        def generate(self, *a, **kw):
            return mdl.generate(*a, **kw)

    adapter = _GPT2Adapter(mdl)

    # Pick a new probe (index 50 — the FIRST new probe past the q_test prefix;
    # this is the exact slot where the round-4 run KeyErrored).
    first_new_probe = probes_500[50]
    # Defensive: the index-50 entry MUST equal new_probes_450[0] by spec.
    assert first_new_probe == new_probes_450[0], (
        f"probes_500[50] != new_probes_450[0]: {first_new_probe!r} vs "
        f"{new_probes_450[0]!r} — q_test prefix invariant violated."
    )

    # Build the actual D1 Condition from the i406 catalog (NOT a stub —
    # we want the real Class-D code path with register='formal').
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

    d1 = CONDITIONS_BY_ID["D1"]
    assert d1.cls == "D" and d1.register == "formal", (
        f"Expected D1 to be Class-D formal-register; got cls={d1.cls!r} register={d1.register!r}"
    )

    target_layers = (0, 1)
    extraction_points = ("last_prompt",)  # mean_response would need a generate; lp is enough

    # B=1 single-probe extract using the REAL Class-D code path through
    # _build_prompts_for_extraction. THIS is the gate: if class_d_rewrites
    # didn't cover this probe, the next 3 lines KeyError.
    with _LayerHookCapture(adapter, target_layers) as cap:
        rows_b, _meta_b, _, _ = _extract_batch(
            adapter,
            tok,
            device="cpu",
            cond=d1,
            questions=[first_new_probe],
            class_d_rewrites=merged,
            extraction_points=extraction_points,
            layers=target_layers,
            max_response_tokens=2,
            hook_capture=cap,
            capture_next_token_logits=False,
        )

    # Validate we got a non-NaN extraction.
    for L in target_layers:
        v = rows_b[0]["last_prompt"][L]
        assert torch.is_tensor(v), f"expected tensor at layer {L}, got {type(v).__name__}"
        assert not torch.any(torch.isnan(v)), (
            f"Class-D extraction on new probe layer {L} returned all-NaN"
        )

    return {
        "extension_path": str(ext_path),
        "extension_kind": ext_kind,
        "n_probes_in_pool": len(probes_500),
        "n_in_merged_rewrites": len(merged),
        "n_new_probes_covered": 450 - len(missing_new),
        "registers_per_probe": list(REGISTERS),
        "integration_probe": first_new_probe[:60],
        "integration_cond": d1.cid,
        "integration_register": d1.register,
        "integration_extracted_layers": list(target_layers),
    }


# ──────────── Check 19: dispatcher overrides stale Class-D env ─────────────


def check_dispatcher_overrides_stale_class_d_env() -> dict:
    """Round-6 regression: the dispatcher MUST unconditionally override any
    stale ``EPM_CLASS_D_REWRITES_EXTENSION_PATH`` inherited from its parent
    shell. The round-5 implementation guarded the set with
    ``if "EPM_..." not in os.environ:`` — so on a RESUMED pod that exported
    a now-stale path, the stale value would survive into workers, and
    ``load_class_d_rewrites`` would ``FileNotFoundError`` before any extract.

    The check pre-sets a stale path in this process's env, calls
    ``_set_class_d_env_var`` with a real existing path, and asserts the
    real path won (string equality on the resolved absolute path). It also
    exercises the "missing extension" branch to confirm the function does
    NOT set the env var when the path is absent — that case is the
    fail-fast gate's responsibility, not the env-set's.

    Restores the original env var afterwards (test hygiene).
    """
    import os as _os
    from pathlib import Path as _Path

    # Lazy-import the dispatcher's env-set helper. The dispatcher imports
    # heavy modules at top of file (subprocess + argparse); both are stdlib
    # so the import is cheap on the dev VM (no torch / vllm).
    from issue502_dispatch import _set_class_d_env_var

    STALE = "/nonexistent/stale-must-not-survive.json"
    ENV_KEY = "EPM_CLASS_D_REWRITES_EXTENSION_PATH"

    # Pick a real, existing extension JSON: prefer the real artefact, fall
    # back to the smoke one materialized by check 18. Either is a valid
    # file on disk for this test's purposes.
    REAL_EXT = PROJECT_ROOT / "eval_results" / "issue_502" / "class_d_rewrites_extended_v1.json"
    SMOKE_EXT = (
        PROJECT_ROOT / "eval_results" / "issue_502" / "class_d_rewrites_extended_v1.smoke.json"
    )
    if REAL_EXT.exists():
        real_path = REAL_EXT
        ext_kind = "real"
    elif SMOKE_EXT.exists():
        real_path = SMOKE_EXT
        ext_kind = "smoke"
    else:
        raise FileNotFoundError(
            f"Neither {REAL_EXT} nor {SMOKE_EXT} exists; run "
            "check_class_d_rewrites_coverage_full_probe_pool first so the "
            "smoke extension materializes."
        )

    expected_resolved = str(_Path(real_path).resolve())
    prior_env = _os.environ.get(ENV_KEY)
    try:
        # CASE 1: stale value pre-set → dispatcher must override it.
        _os.environ[ENV_KEY] = STALE
        assert _os.environ[ENV_KEY] == STALE, "test setup: stale not staged"

        returned = _set_class_d_env_var(real_path)

        assert _os.environ[ENV_KEY] != STALE, (
            f"REGRESSION: stale {ENV_KEY}={STALE!r} survived the dispatcher's "
            "env-set. The conditional guard `if EPM_... not in os.environ` "
            "was reintroduced. Workers would inherit the stale path and "
            "crash at extract time."
        )
        assert _os.environ[ENV_KEY] == expected_resolved, (
            f"{ENV_KEY} = {_os.environ[ENV_KEY]!r}, expected resolved real "
            f"path {expected_resolved!r}"
        )
        assert returned == expected_resolved, (
            f"_set_class_d_env_var returned {returned!r}, expected {expected_resolved!r}"
        )

        # CASE 2: no stale, missing path (None) → env var must NOT be set.
        # The fail-fast gate in main() handles the "missing and required"
        # case; the env-set helper only sets when the path exists.
        _os.environ.pop(ENV_KEY, None)
        returned_none = _set_class_d_env_var(None)
        assert returned_none is None, (
            f"_set_class_d_env_var(None) returned {returned_none!r}, expected None"
        )
        assert ENV_KEY not in _os.environ, (
            f"{ENV_KEY} was set to {_os.environ.get(ENV_KEY)!r} despite "
            "the input path being None — env-set should be a no-op."
        )

        # CASE 3: nonexistent-but-not-None path → env var must NOT be set.
        # Same fail-fast-gate-handles-it rationale as CASE 2.
        bogus = _Path("/nonexistent/never-existed.json")
        returned_bogus = _set_class_d_env_var(bogus)
        assert returned_bogus is None, (
            f"_set_class_d_env_var(<bogus>) returned {returned_bogus!r}, expected None"
        )
        assert ENV_KEY not in _os.environ, (
            f"{ENV_KEY} was set despite the path not existing on disk."
        )
    finally:
        # Restore the original env var exactly.
        if prior_env is None:
            _os.environ.pop(ENV_KEY, None)
        else:
            _os.environ[ENV_KEY] = prior_env

    return {
        "stale_overridden": True,
        "real_path_kind": ext_kind,
        "real_path_resolved": expected_resolved,
        "none_input_is_noop": True,
        "missing_path_is_noop": True,
    }


# ──────── Check 20: regress phase skips the cross_check_406 sidecar ─────────


def check_regress_skips_cross_check_sidecar() -> dict:
    """Round-8 regression: ``_enumerate_predictors`` MUST skip the
    ``__cross_check_406.json`` sidecar that ``write_next_token_js_matrix``
    drops into ``METRIC_DIR``. The sidecar carries a cross-check schema
    (``{schema_version, failed, summary?/failure_reason?, git_sha,
    timestamp_utc}``) — no ``extraction_point`` field — so reading it as
    a predictor crashed the regress phase with ``KeyError:
    'extraction_point'`` (#502 round-7 pod-side launch).

    The check materializes a tmp directory containing:
      (a) one REAL distance-matrix file (``last_prompt__layer5__cosine__raw.json``),
      (b) the cross-check sidecar shape this bug bit on,
      (c) for thoroughness, a ``__perm.json`` MMD permutation companion
          which the prior round also skips.

    It calls ``_enumerate_predictors`` on the sorted file list and
    asserts:
      - the returned row count equals 1 (sidecars are excluded);
      - the single row's ``file`` is the real predictor;
      - no ``KeyError`` is raised.

    Naming any future schema-different sidecar that lands in METRIC_DIR
    must either follow the ``__perm`` / ``__cross_check_406`` skip-pattern
    or carry the full distance-matrix schema. The skip is intentionally
    name-pattern-based for explicitness over a `extraction_point in
    payload` schema sniff — silent skip of an unexpected schema mismatch
    would be the more dangerous failure mode.
    """
    import json as _json
    import tempfile as _tmp
    from pathlib import Path as _Path

    from issue493_extraction_metric_bakeoff import _enumerate_predictors

    tmp_dir = _Path(_tmp.mkdtemp(prefix="i502_smoke_enum_predictors_"))

    # (a) Real distance-matrix file — full predictor schema.
    real_payload = {
        "schema_version": 1,
        "extraction_point": "last_prompt",
        "layer": 5,
        "metric": "cosine",
        "variant": "raw",
        "pca_k": 16,
        "cond_ids": ["A1", "A2"],
        "matrix": {"A1": {"A1": 0.0, "A2": 0.1}, "A2": {"A1": 0.1, "A2": 0.0}},
        "git_sha": "test",
        "timestamp_utc": "now",
    }
    real_path = tmp_dir / "last_prompt__layer5__cosine__raw.json"
    real_path.write_text(_json.dumps(real_payload))

    # (b) Cross-check sidecar — schema-different, NO `extraction_point`.
    # Exact shape produced by `write_next_token_js_matrix` on the
    # success branch (the bug also fired on the failure branch which
    # has identical key set minus `summary`).
    sidecar_payload = {
        "schema_version": 1,
        "failed": False,
        "summary": {"rank_corr": 0.97},
        "git_sha": "test",
        "timestamp_utc": "now",
    }
    sidecar_path = tmp_dir / "last_prompt__layer-1__next_token_js__raw__cross_check_406.json"
    sidecar_path.write_text(_json.dumps(sidecar_payload))

    # (c) MMD permutation companion — already-handled prior skip pattern,
    # included for thoroughness so this single check guards both
    # name-pattern filters.
    perm_payload = {
        "schema_version": 1,
        "extraction_point": "last_prompt",
        "layer": 5,
        "metric": "mmd",
        "variant": "raw",
        "n_perm": 100,
        "summary": {},
        "git_sha": "test",
        "timestamp_utc": "now",
    }
    perm_path = tmp_dir / "last_prompt__layer5__mmd__raw__perm.json"
    perm_path.write_text(_json.dumps(perm_payload))

    all_files = sorted(tmp_dir.glob("*.json"))
    assert len(all_files) == 3, f"test setup: expected 3 tmp files, got {len(all_files)}"

    # The bug under test: pre-round-8, this call raised
    # ``KeyError: 'extraction_point'`` on the sidecar payload. With the
    # fix it MUST return exactly one row, derived from the real file.
    rows = _enumerate_predictors(all_files)

    assert len(rows) == 1, (
        f"_enumerate_predictors returned {len(rows)} rows; expected exactly 1 "
        "(the real distance-matrix file). Sidecars (__perm, __cross_check_406) "
        "must be skipped."
    )
    row = rows[0]
    assert row["file"] == str(real_path), (
        f"_enumerate_predictors returned the wrong file: got {row['file']!r}, "
        f"expected {str(real_path)!r}"
    )
    assert row["extraction_point"] == "last_prompt"
    assert row["metric"] == "cosine"

    return {
        "input_files": len(all_files),
        "predictor_rows": len(rows),
        "sidecar_skipped": True,
        "perm_skipped": True,
        "predictor_file": real_path.name,
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
        # Round-4 additions
        ("prod_js_cross_check_sidecar_asserted", check_prod_js_cross_check_sidecar_asserted),
        ("cache_bypass_stale_canonical_raises", check_cache_bypass_stale_canonical_raises),
        ("validate_canonical_completeness_gate", check_validate_canonical_completeness_gate),
        # Round-5 addition (the round-4 bounce: Class-D KeyError on new probes)
        (
            "class_d_rewrites_coverage_full_probe_pool",
            check_class_d_rewrites_coverage_full_probe_pool,
        ),
        # Round-6 addition: dispatcher must override stale Class-D env var
        # (regression test for the round-5 conditional-guard bug).
        (
            "dispatcher_overrides_stale_class_d_env",
            check_dispatcher_overrides_stale_class_d_env,
        ),
        # Round-8 addition: regress phase must skip the cross_check_406
        # sidecar in METRIC_DIR (regression test for the round-7
        # KeyError:'extraction_point' crash at the regress entry).
        (
            "regress_skips_cross_check_sidecar",
            check_regress_skips_cross_check_sidecar,
        ),
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
