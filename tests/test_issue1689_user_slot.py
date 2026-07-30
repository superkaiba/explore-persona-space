"""Tests for the #1689 ``user-slot-recapture`` round (render / capture / fits).

Pins, in order of the failure they prevent:

* the naturalistic boundary layout is NON-DEGENERATE (the realized defect this
  round exists to fix: the parent's user naturalistic cells had
  prefix_end == context_end == answer_end, so X_prefix == X_context == Y and
  identity+bias R2 was 1.0000 exactly);
* chat offsets are anchored on the CONTENT-INDEPENDENT template tail, never a
  ``text.find(content)`` scan (#1776: a short query substring-matches inside
  Qwen's default-system preamble and silently yields garbage spans);
* the straddler policy is exercised in BOTH directions on a real BPE merge;
* the reduced-basis companion's truncation identity holds and genuinely bites;
* the parent row-set reconstruction (``expand_by_dup``) is exact;
* Gate-1 is NOT inert — a wrong published reference must raise;
* a tiny-real CPU end-to-end capture produces a store whose slot vectors are
  genuinely DISTINCT (the direct regression pin on the collapse defect), using
  the REAL tokenizer and faking ONLY the GPU-scale weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.issue1689_user_slot_capture import (
    assign_units_to_gpus,
    resolve_slot_token,
    run_worker,
)
from scripts.issue1689_user_slot_fits import (
    expand_by_dup,
    gate1_parent_parity,
    verify_truncation_equivalence,
)
from scripts.issue1689_user_slot_render import (
    LMSYS_CONST_U2,
    SLOT_STRADDLER_POLICY,
    STORY_USER_LABEL_TEMPLATE,
    STORY_USER_TEMPLATE,
    UNIT_BY_ID,
    build_units,
    naturalistic_text_and_offsets,
    story_text_and_offsets,
)

REAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _tokenizer():
    """Real Qwen tokenizer, or skip loudly (offline machine / cold HF cache)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(REAL_MODEL)
    except Exception as exc:
        pytest.skip(f"real tokenizer unavailable ({type(exc).__name__}: {exc})")


# ---------------------------------------------------------------------------
# Render: layout
# ---------------------------------------------------------------------------


def test_unit_lattice_is_24_unique_cells():
    units = build_units()
    assert len(units) == 24
    assert len({u.unit_id for u in units}) == 24
    story = [u for u in units if u.framing == "story"]
    assert {u.variant for u in story} == {"alex", "user_label"}
    assert len(story) == 12


def test_naturalistic_offsets_strictly_increasing_and_exact():
    """The realized-defect regression pin: no two boundaries may coincide."""
    u1, a1, u2 = "hello there", "sure, here you go", "and what about tomorrow?"
    text, off = naturalistic_text_and_offsets(u1, a1, u2)
    order = ["first_user_header_end", "u1_end", "prev_turn_end", "u2_header_end", "u2_end"]
    vals = [off[k] for k in order]
    assert vals == sorted(vals) and len(set(vals)) == len(vals), off
    # Boundaries land exactly where the docstring says.
    assert text[: off["first_user_header_end"]] == "User: "
    assert text[off["first_user_header_end"] : off["u1_end"]] == u1
    assert text[off["u1_end"] : off["prev_turn_end"]] == f"\n\nAssistant: {a1}\n\n"
    assert text[off["prev_turn_end"] : off["u2_header_end"]] == "User: "
    assert text[off["u2_header_end"] : off["u2_end"]] == u2
    assert off["u2_end"] == len(text)


def test_story_offsets_and_label_variant_is_minimal_substitution():
    assert STORY_USER_TEMPLATE.count("Alex") == 3
    assert STORY_USER_TEMPLATE.replace("Alex", "User") == STORY_USER_LABEL_TEMPLATE
    assert "Alex" not in STORY_USER_LABEL_TEMPLATE
    u1, a1, u2 = "q one", "a one", "q two"
    for variant in ("alex", "user_label"):
        text, off = story_text_and_offsets(u1, a1, u2, variant=variant)
        vals = [
            off[k] for k in ("prev_turn_end", "story_prefix_end", "u2_end", "parent_answer_end")
        ]
        assert vals == sorted(vals) and len(set(vals)) == len(vals), (variant, off)
        # story_prefix_end sits immediately after u2's OPEN QUOTE (parent convention)
        assert text[off["story_prefix_end"] - 1] == '"'
        assert text[off["story_prefix_end"] : off["u2_end"]] == u2
        assert off["parent_answer_end"] == len(text)


def test_lmsys_constant_u2_is_the_parent_fallback_string():
    """The parent corpus has no u2_lmsys, so this exact 34-char constant IS the
    realized lmsys u2 — the round carries it as a labelled constant-u2 control."""
    assert LMSYS_CONST_U2 == "Can you say a bit more about that?"
    assert len(LMSYS_CONST_U2) == 34


def test_chat_offsets_anchor_on_template_tail_not_content_find():
    """A 1-char u2 that also occurs inside the template preamble must still
    resolve to the LAST user turn (#1776 mis-anchor class)."""
    from scripts.issue1689_user_slot_render import chat_text_and_offsets

    tok = _tokenizer()
    u1, a1 = "what is 2+2?", "4"
    for u2 in ("a", ".", "You", "and then?"):
        text, off = chat_text_and_offsets(u1, a1, u2, tok)
        assert text[off["u2_header_end"] : off["u2_end"]] == u2
        assert (
            off["u2_header_end"]
            > off["prev_turn_end"]
            > off["u1_end"]
            > off["first_user_header_end"]
            > 0
        ), off
        assert text[off["first_user_header_end"] : off["u1_end"]] == u1
        assert text.endswith("\n")
        assert off["parent_answer_end"] == len(text)
        # The u2 header must be a real chat header, not a preamble hit.
        assert text[: off["u2_header_end"]].endswith("<|im_start|>user\n")


# ---------------------------------------------------------------------------
# Capture: slot resolution
# ---------------------------------------------------------------------------


def test_resolve_slot_token_straddler_both_directions():
    # token spans: [0,3) [3,7) [7,10)
    offsets = [(0, 3), (3, 7), (7, 10)]
    # boundary at a token edge: both policies agree, no straddle
    assert resolve_slot_token(offsets, 7, straddler_include=False) == (1, False)
    assert resolve_slot_token(offsets, 7, straddler_include=True) == (1, False)
    # boundary INSIDE token 1: exclude drops it (and flags), include keeps it
    assert resolve_slot_token(offsets, 5, straddler_include=False) == (0, True)
    assert resolve_slot_token(offsets, 5, straddler_include=True) == (1, False)
    with pytest.raises(ValueError):
        resolve_slot_token(offsets, 0, straddler_include=False)


def test_naturalistic_label_boundary_really_straddles_on_real_bpe():
    """The ``User: `` label's trailing space merges into u2's first word on a
    real Qwen tokenizer, so the exclude policy MUST report a straddle here —
    the #1315 plain-text-boundary class, expected-dense rather than anomalous."""
    tok = _tokenizer()
    text, off = naturalistic_text_and_offsets("hi", "hello", "How are you today?")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    _, straddled = resolve_slot_token(spans, off["u2_header_end"], straddler_include=False)
    assert straddled, "expected the label/u2 boundary to straddle a merged token"
    # The Y-side end-of-content slot must NOT be reported as an excluded straddler.
    _, y_straddled = resolve_slot_token(spans, off["u2_end"], straddler_include=True)
    assert not y_straddled


def test_slot_straddler_policy_covers_every_declared_slot():
    for u in build_units():
        for s in u.slots:
            assert s in SLOT_STRADDLER_POLICY, s


def test_hub_helper_call_shapes_bind():
    """Signature-bind every hub helper this round calls from an upload/staging
    branch a CPU smoke cannot reach.

    Import resolution (and module-top hoisting) both green-light an
    arity/keyword mismatch, which then fires on the pod at the TERMINAL upload
    stage (#1332). ``verify_repo_paths_uploaded``'s ``path_in_repo`` is required
    KEYWORD-ONLY and was in fact missing from the first draft of this round's
    uploader — caught by a live probe, pinned here.
    """
    import inspect

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        list_hf_files_under_path,
        retry_transient,
        stage_hub_file,
        verify_repo_paths_uploaded,
    )

    inspect.signature(verify_repo_paths_uploaded).bind(
        object(), "repo", ["p"], path_in_repo="prefix", repo_type="dataset"
    )
    with pytest.raises(TypeError):
        inspect.signature(verify_repo_paths_uploaded).bind(
            object(), "repo", ["p"], repo_type="dataset"
        )
    inspect.signature(retry_transient).bind(lambda: None, what="x")
    inspect.signature(stage_hub_file).bind("repo", "path", "target", revision="r")
    inspect.signature(list_hf_files_under_path).bind(
        object(), "repo", "prefix", repo_type="dataset", revision="r"
    )
    inspect.signature(HfApi.upload_folder).bind(
        object(),
        folder_path="f",
        path_in_repo="p",
        repo_id="r",
        repo_type="dataset",
        allow_patterns=["**/*.pt"],
    )


def test_assign_units_to_gpus_covers_all_units_and_balances():
    entries = [
        {
            "unit_id": f"u{i}",
            "model": "A" if i % 2 else "B",
            "n_rows": 100 * (i + 1),
            "token_len_p50": 50,
        }
        for i in range(8)
    ]
    plan = assign_units_to_gpus(entries, 4)
    assigned = [u for v in plan.values() for u in v]
    assert sorted(assigned) == sorted(e["unit_id"] for e in entries)
    assert len(plan) == 4
    # one model per GPU: no GPU may serve both models
    model_of = {e["unit_id"]: e["model"] for e in entries}
    for units in plan.values():
        assert len({model_of[u] for u in units}) <= 1
    with pytest.raises(RuntimeError):
        assign_units_to_gpus(entries, 0)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def test_expand_by_dup_reconstructs_a_parent_row_set():
    X = np.arange(6, dtype=np.float32).reshape(3, 2)
    dup = np.array([3, 1, 2])
    out = expand_by_dup(X, dup)
    assert out.shape == (6, 2)
    assert np.array_equal(out[:3], np.repeat(X[:1], 3, axis=0))
    ids = expand_by_dup(np.array(["a", "b", "c"], dtype=object), dup)
    assert list(ids) == ["a", "a", "a", "b", "c", "c"]


def test_truncation_equivalence_holds_and_bites():
    out = verify_truncation_equivalence()
    assert out["pass"]
    assert out["max_abs_at_full_k"] <= out["tol"]
    assert out["delta_at_half_k"] > out["bite_floor"]
    assert out["n_train"] < out["d"], "fixture must be in the production rank regime"


def test_fit_grid_shares_x_factorizations_and_covers_all_combos():
    """The X-side factorization is built ONCE per (X, fold) and reused by every
    Y variant.

    This sharing is load-bearing: six independent combos would build 6 x N_FOLDS
    factorizations per group instead of 2 x N_FOLDS, which projects the grid
    battery to ~145 h at production shape. The audit counters make a silent
    regression to the unshared shape impossible to miss.
    """
    from scripts.issue1689_user_slot_fits import N_FOLDS, fit_grid

    # Production rank regime: n_train (48) < d, so the Gram is full rank and the
    # reduced basis k = n_train // 2 genuinely truncates. A fixture with d <
    # n_train keeps every nonzero eigenvalue and makes the companion INERT.
    n, d = 60, 200
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((n, d))
    kinds = ("X_clean", "X_straddle", "Y_mean", "Y_end", "Y_boundary")
    grid = {
        k: (latent @ rng.standard_normal((d, d)) * 0.4 + rng.standard_normal((n, d))).astype(
            np.float32
        )
        for k in kinds
    }
    ids = np.array([f"c{i // 2}" for i in range(n)], dtype=object)
    x_kinds = ("X_clean", "X_straddle")
    y_kinds = ("Y_mean", "Y_end", "Y_boundary")
    out = fit_grid(grid, ids, x_kinds=x_kinds, y_kinds=y_kinds, null_draws=4)

    assert out["shared_factorizations"] == len(x_kinds) * N_FOLDS
    assert out["unshared_would_have_been"] == len(x_kinds) * N_FOLDS * len(y_kinds)
    assert out["shared_factorizations"] * len(y_kinds) == out["unshared_would_have_been"]
    assert set(out["combos"]) == {f"{x}->{y}" for x in x_kinds for y in y_kinds}
    for key, row in out["combos"].items():
        assert np.isfinite(row["r2"]), key
        assert np.isfinite(row["r2_reduced_basis"]), key
        assert row["null_shuffle_fit_targets"]["n_draws"] == 4, key
        # The reduced basis genuinely bites: k = n_train // 2 discards real
        # eigen-directions because n_train < d makes the Gram full rank.
        assert row["r2_reduced_basis"] != row["r2"], key
        assert "identity_bias_r2" in row, key
        assert row["knn_cosine"]["n_pool"] > 0, key


def test_gate1_raises_on_a_wrong_published_reference(tmp_path: Path):
    """Gate-1 non-inertness: a mismatching reference MUST fail loud."""
    n, d = 40, 8
    rng = np.random.default_rng(0)
    store = {
        "slots": {
            "prev_turn_end": rng.standard_normal((n, d)).astype(np.float32),
            "parent_answer_end": rng.standard_normal((n, d)).astype(np.float32),
        },
        "conv_ids": np.array([f"c{i // 2}" for i in range(n)], dtype=object),
        "dup_count": np.ones(n, dtype=np.int32),
        "slot_names": ["prev_turn_end", "parent_answer_end"],
    }
    stores = {
        "Qwen_Qwen2.5-7B__chat__lmsys": store,
        "Qwen_Qwen2.5-7B-Instruct__chat__lmsys": store,
    }
    for m in ("Qwen_Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct"):
        (tmp_path / f"heldout_r2_{m}_user_lmsys_chat.json").write_text(
            json.dumps(
                {
                    "n_rows": n,
                    "layers": [14, 18, 19, 26],
                    "prefix": {"held_out_r2_per_layer": [0.0, 0.0, -99.0, 0.0]},
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError, match="Gate-1 FAILED"):
        gate1_parent_parity(stores, tmp_path)


def test_gate1_raises_on_missing_reference(tmp_path: Path):
    with pytest.raises(RuntimeError, match="published reference missing"):
        gate1_parent_parity(
            {
                "Qwen_Qwen2.5-7B__chat__lmsys": {
                    "slots": {},
                    "conv_ids": np.array([], dtype=object),
                    "dup_count": np.array([], dtype=np.int32),
                    "slot_names": [],
                }
            },
            tmp_path,
        )


# ---------------------------------------------------------------------------
# Tiny-real CPU end-to-end capture
# ---------------------------------------------------------------------------


def _build_tiny_qwen2(dest: Path, tokenizer) -> Path:
    """A REAL Qwen2 architecture with >=20 blocks at toy width.

    ``hidden_states[19]`` must exist, so the stub keeps 20 layers and shrinks
    only the widths — the sole fake in the tiny-real run is the GPU-scale
    weights, per the tiny-real recipe.
    """
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=20,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(dest)
    return dest


def test_tiny_real_cpu_capture_end_to_end(tmp_path: Path):
    """Run the production worker path on CPU and assert the slots are DISTINCT.

    This is the direct regression pin on the collapse defect: a store whose
    ``prev_turn_end`` / ``u2_header_end`` / ``u2_end`` vectors are equal is the
    bug this round fixes, so the assertion is that they differ.
    """
    import argparse
    from dataclasses import asdict

    import torch

    from scripts.issue1689_user_slot_render import (
        FIT_PAIRS_BY_FRAMING,
        GRID_SLOT_KINDS,
        PRIMARY_FIT_BY_FRAMING,
        SLOTS_BY_FRAMING,
        build_read_groups,
        chat_text_and_offsets,
        sha256_text,
    )

    tok = _tokenizer()
    weights = _build_tiny_qwen2(tmp_path / "tiny", tok)

    rendered = tmp_path / "rendered"
    rendered.mkdir()
    units = []
    for framing in ("chat", "naturalistic"):
        unit_id = f"Qwen_Qwen2.5-7B-Instruct__{framing}__onpolicy"
        rows = []
        for i in range(4):
            u1, a1, u2 = f"question {i} about things", f"answer {i}", f"follow up {i} please"
            if framing == "chat":
                text, off = chat_text_and_offsets(u1, a1, u2, tok)
            else:
                text, off = naturalistic_text_and_offsets(u1, a1, u2)
            unit = UNIT_BY_ID[f"Qwen_Qwen2.5-7B-Instruct__{framing}__onpolicy"]
            text, groups = build_read_groups(unit, off, text)
            rows.append(
                {
                    "unit_id": unit_id,
                    "row_index": i,
                    "conv_id": f"c{i}",
                    "dup_count": 1,
                    "u2_provenance": "onpolicy",
                    "judge_score_mean": 90.0,
                    "text": text,
                    "char_slots": off,
                    "read_groups": [asdict(g) for g in groups],
                    "n_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                    "text_sha256": sha256_text(text),
                }
            )
        (rendered / f"{unit_id}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        units.append(
            {
                "unit_id": unit_id,
                "model": REAL_MODEL,
                "model_dir": "Qwen_Qwen2.5-7B-Instruct",
                "framing": framing,
                "provenance": "onpolicy",
                "variant": "base",
                "slots": list(SLOTS_BY_FRAMING[framing]),
                "fit_pairs": [list(p) for p in FIT_PAIRS_BY_FRAMING[framing]],
                "primary_fit": PRIMARY_FIT_BY_FRAMING[framing],
                "rendered_path": f"{unit_id}.jsonl",
                "n_rows": len(rows),
            }
        )
    manifest = {"metadata": {}, "units": units}
    (rendered / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out_root = tmp_path / "store"
    args = argparse.Namespace(
        units=",".join(u["unit_id"] for u in units),
        gpu_id=0,
        rendered_dir=rendered,
        out_root=out_root,
        batch_size=2,
        max_rows=0,
        overwrite=True,
        allow_cpu=True,
        weights_override=str(weights),
    )
    assert run_worker(args, manifest) == 0

    for u in units:
        path = out_root / u["model_dir"] / u["unit_id"] / "L19.pt"
        assert path.exists(), path
        st = torch.load(path, map_location="cpu", weights_only=False)
        assert st["layer"] == 19
        assert st["n_rows"] == 4
        assert sorted(st["slot_names"]) == sorted(u["slots"])
        for s in u["slots"]:
            arr = st["slots"][s]
            assert arr.shape == (4, 64), (s, arr.shape)
            assert np.isfinite(arr).all()
        # THE regression pin: slot vectors must not be bit-identical.
        names = list(u["slots"])
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = st["slots"][names[i]], st["slots"][names[j]]
                assert not np.array_equal(a, b), (
                    f"{u['unit_id']}: slots {names[i]} and {names[j]} are bit-identical — "
                    "the collapse defect has returned"
                )
        # token positions strictly ordered, matching the char-offset order
        pos = [int(st["slot_token_pos"][s][0]) for s in names]
        assert pos == sorted(pos) and len(set(pos)) == len(pos), dict(zip(names, pos, strict=True))
        # --- addendum E: the X x Y grid -------------------------------------
        gnames = list(st["grid_group_names"])
        assert "u2" in gnames and "u1" in gnames, gnames
        for gn in gnames:
            for kind in GRID_SLOT_KINDS:
                arr = st["grid_slots"][f"{gn}__{kind}"]
                assert arr.shape == (4, 64), (gn, kind, arr.shape)
                assert np.isfinite(arr).all(), (gn, kind)
            gp = st["grid_slot_pos"]
            xc = int(gp[f"{gn}__X_clean"][0])
            xs = int(gp[f"{gn}__X_straddle"][0])
            ye = int(gp[f"{gn}__Y_end"][0])
            yb = int(gp[f"{gn}__Y_boundary"][0])
            assert xc < xs <= ye <= yb, (gn, xc, xs, ye, yb)
            assert xs == xc + 1, (gn, xc, xs)
            lo, hi = (int(v) for v in st["grid_answer_span"][gn][0])
            assert (lo, hi) == (xs, ye), (gn, lo, hi, xs, ye)
            # X_clean and X_straddle are DISTINCT positions, so their vectors
            # must differ — the straddler-exclusive/inclusive contrast is real.
            assert not np.array_equal(
                st["grid_slots"][f"{gn}__X_clean"], st["grid_slots"][f"{gn}__X_straddle"]
            ), gn
            # Y_mean over a >1-token span is not any single stored position.
            if hi > lo:
                assert not np.array_equal(
                    st["grid_slots"][f"{gn}__Y_mean"], st["grid_slots"][f"{gn}__Y_end"]
                ), gn
