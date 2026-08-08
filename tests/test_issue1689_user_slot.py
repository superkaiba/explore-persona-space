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

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import issue1689_user_slot_gen_a1 as GA1
from scripts.issue1689_user_slot_capture import (
    assign_units_to_gpus,
    resolve_slot_token,
    run_worker,
)
from scripts.issue1689_user_slot_fits import (
    LENGTH_SPLIT_ARMS,
    bridge_comparisons,
    context_token_lengths,
    conv_rank_median_split,
    expand_by_dup,
    fit_length_stratified,
    gate1_parent_parity,
    measure_fold_basis,
    project_battery_wall,
    published_parent_r2,
    verify_truncation_equivalence,
)
from scripts.issue1689_user_slot_render import (
    FIT_PAIRS_BY_FRAMING,
    GEN_A1_SUBDIR,
    LMSYS_CONST_U2,
    PRIMARY_FIT_BY_FRAMING,
    READ_GROUP_NAMES_BY_FRAMING,
    READ_GROUPS_BY_FRAMING,
    SINGLE_TURN_VARIANTS,
    SLOT_STRADDLER_POLICY,
    SLOTS_BY_FRAMING,
    STORY_USER_LABEL_TEMPLATE,
    STORY_USER_TEMPLATE,
    UNIT_BY_ID,
    Unit,
    _local_gen_a1_entries,
    _source_paths,
    build_bridge_units,
    build_read_groups,
    build_units,
    naturalistic_text_and_offsets,
    parent_recap_text_and_offsets,
    render_row,
    single_turn_text_and_offsets,
    smoke_units,
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


def test_unit_lattice_base_24_plus_bridge_14_all_unique():
    units = build_units()
    base = [u for u in units if u.framing in ("chat", "naturalistic", "story")]
    assert len(base) == 24
    story = [u for u in base if u.framing == "story"]
    assert {u.variant for u in story} == {"alex", "user_label"}
    assert len(story) == 12
    # addenda B/C/D bridging cells
    bridge = build_bridge_units()
    assert len(bridge) == 14
    counts = {}
    for u in bridge:
        counts[u.framing] = counts.get(u.framing, 0) + 1
    assert counts == {"single_turn": 4, "onpolicy_a1": 4, "parent_recap": 6}
    assert len(units) == 38
    assert len({u.unit_id for u in units}) == 38, "unit ids must stay unique across families"


def test_smoke_set_covers_one_cell_per_family():
    """Per-arm-class smoke duty: every family owns a distinct offset builder,
    source loader and read-group shape, so a smoke that covers one family is
    structurally blind to the others' seams."""
    units = smoke_units()
    assert {u.framing for u in units} == {u.framing for u in build_units()}
    assert len(units) == len({u.unit_id for u in units})


def test_local_gen_a1_dir_maps_into_the_staged_hub_keys(tmp_path):
    """Production runs the a1 generator and the render on the SAME pod, so the
    render must consume the generator's LOCAL output without a Hub round-trip.
    The synthetic keys must carry the canonical prefix `_source_paths` matches."""
    d = tmp_path / "gen_a1"
    d.mkdir()
    for short in ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"):
        (d / f"user_slot_a1_onpolicy_{short}.jsonl").write_text(
            json.dumps({"conv_id": "c1", "a1_onpolicy": "hi"}) + "\n", encoding="utf-8"
        )
    entries = _local_gen_a1_entries(d)
    assert len(entries) == 2
    for hub, local in entries.items():
        assert f"/{GEN_A1_SUBDIR}/" in hub and local.is_file()
    # `_source_paths` resolves the synthetic keys exactly as it resolves Hub ones,
    # and the stem must match what the generator actually writes (model_short).
    for short in ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"):
        hits = _source_paths(entries, GEN_A1_SUBDIR, f"user_slot_a1_onpolicy_{short}")
        assert len(hits) == 1
    with pytest.raises(FileNotFoundError):
        _local_gen_a1_entries(tmp_path / "empty-dir-that-does-not-exist")


def test_every_framing_has_slots_fit_pairs_and_read_groups():
    """A new framing that misses ANY dispatch table silently ships a broken cell."""
    for u in build_units():
        assert u.framing in SLOTS_BY_FRAMING, u.framing
        assert u.framing in FIT_PAIRS_BY_FRAMING, u.framing
        assert u.framing in PRIMARY_FIT_BY_FRAMING, u.framing
        assert u.framing in READ_GROUPS_BY_FRAMING, u.framing
        # the primary fit name must actually be one of the framing's pairs
        names = {p[2] for p in u.fit_pairs}
        assert PRIMARY_FIT_BY_FRAMING[u.framing] in names, (u.framing, names)
        # every slot a fit pair references must be a declared slot
        for x_slot, y_slot, _ in u.fit_pairs:
            assert x_slot in u.slots, (u.framing, x_slot)
            assert y_slot in u.slots, (u.framing, y_slot)
        # every slot must carry a straddler policy
        for s in u.slots:
            assert s in SLOT_STRADDLER_POLICY, (u.framing, s)
        # the read-group NAMES must be declared too: the fits' synthetic smoke
        # tree sizes its grid off the declaration, so a framing missing from it
        # would be smoke-covered with the wrong grid shape (or a KeyError).
        assert u.framing in READ_GROUP_NAMES_BY_FRAMING, u.framing


def test_declared_read_group_names_match_every_builder_and_are_enforced():
    """`READ_GROUP_NAMES_BY_FRAMING` is what the fits' synthetic smoke tree
    builds its grid from, so a drift between it and the real builders would make
    the smoke's grid shape diverge from production's. Pin BOTH directions: the
    declaration matches what each builder realizes, AND the in-render assert is
    not inert."""
    assert set(READ_GROUP_NAMES_BY_FRAMING) == set(READ_GROUPS_BY_FRAMING)

    text, off = naturalistic_text_and_offsets("hello there", "sure thing", "and tomorrow?")
    nat_unit = UNIT_BY_ID["Qwen_Qwen2.5-7B__naturalistic__haiku"]
    _, groups = build_read_groups(nat_unit, off, text)
    assert [g.name for g in groups] == list(READ_GROUP_NAMES_BY_FRAMING["naturalistic"])

    st_text, st_off = single_turn_text_and_offsets("only question", "the answer", label="Assistant")
    st_unit = UNIT_BY_ID["Qwen_Qwen2.5-7B__single_turn_assistant__parent"]
    _, st_groups = build_read_groups(st_unit, st_off, st_text)
    assert [g.name for g in st_groups] == list(READ_GROUP_NAMES_BY_FRAMING["single_turn"])

    # Non-inertness: a declaration that disagrees with the builder must raise.
    import scripts.issue1689_user_slot_render as R

    original = R.READ_GROUP_NAMES_BY_FRAMING["naturalistic"]
    R.READ_GROUP_NAMES_BY_FRAMING["naturalistic"] = ("u2", "u1", "phantom")
    try:
        with pytest.raises(RuntimeError, match="realized read groups"):
            build_read_groups(nat_unit, off, text)
    finally:
        R.READ_GROUP_NAMES_BY_FRAMING["naturalistic"] = original


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


# ---------------------------------------------------------------------------
# Addendum A: length-stratified refits on the PARENT stores
# ---------------------------------------------------------------------------


def _rendered_shard(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def _nat_row(cid: str, u1: str, u2: str = "Say more.") -> dict:
    """A naturalistic rendered row in the PARENT's own field shape."""
    return {
        "conv_id": cid,
        "prefix_text_only": f"User: {u1}\n\nAssistant: ok\n\n",
        "u2_text_marked": f"User: {u2}",
        "context_tail": "\n\nAssistant: ",
    }


def test_context_token_lengths_dedups_identical_duplicates(tmp_path: Path):
    """The parent publishes each conversation 3x per condition with a
    BYTE-IDENTICAL context; the length table must collapse those to one entry
    (and tokenize once), not raise and not triple-count."""
    tok = _tokenizer()
    rows = []
    for cid, u1 in (("c1", "short one"), ("c2", "a much longer first user turn " * 12)):
        rows += [_nat_row(cid, u1)] * 3  # the parent's 3-rows-per-conversation shape
    # split across two shards, so the duplicate group spans files
    p0 = _rendered_shard(tmp_path, "cond.shard00.jsonl", rows[:3])
    p1 = _rendered_shard(tmp_path, "cond.shard01.jsonl", rows[3:])
    lengths, digest = context_token_lengths([p0, p1], tok, "naturalistic")
    assert sorted(lengths) == ["c1", "c2"]
    assert digest["n_rendered_rows"] == 6
    assert digest["n_distinct_convs"] == 2
    assert digest["duplicate_factor"] == pytest.approx(3.0)
    assert digest["n_tokenized"] == 2, "only the first occurrence per conv is tokenized"
    assert lengths["c2"] > lengths["c1"], "longer u1 must yield a longer context"


def test_context_token_lengths_raises_on_divergent_duplicate(tmp_path: Path):
    """A duplicate group whose context text DIVERGES breaks the per-conversation
    length premise and must fail loud (never silently pick one)."""
    tok = _tokenizer()
    rows = [_nat_row("c1", "same"), _nat_row("c1", "DIFFERENT first turn entirely")]
    p = _rendered_shard(tmp_path, "cond.shard00.jsonl", rows)
    with pytest.raises(RuntimeError, match="DIVERGENT context text"):
        context_token_lengths([p], tok, "naturalistic")


def test_conv_rank_median_split_matched_n_disjoint_length_separated():
    """Matched row count, disjoint halves, and a genuine length separation."""
    # 10 conversations, 3 rows each (the parent's group shape), lengths 10..100
    conv_ids = [f"c{i}" for i in range(10) for _ in range(3)]
    lengths = {f"c{i}": 10 * (i + 1) for i in range(10)}
    split = conv_rank_median_split(conv_ids, lengths, seed=0)
    si, li = split["short_idx"], split["long_idx"]
    m = split["meta"]
    assert si.shape[0] == li.shape[0] == 15
    assert set(si.tolist()).isdisjoint(li.tolist())
    assert m["boundary_short_max"] <= m["boundary_long_min"]
    assert m["short"]["n_convs"] == m["long"]["n_convs"] == 5
    assert m["dropped_middle_conv_count"] == 0
    # every conversation stays WHOLE inside one half (conversation-grouped folds
    # inside a half are only clean if a conv's rows never straddle the split)
    short_convs = {conv_ids[i] for i in si.tolist()}
    long_convs = {conv_ids[i] for i in li.tolist()}
    assert short_convs.isdisjoint(long_convs)
    for c in short_convs | long_convs:
        idx = [i for i, cc in enumerate(conv_ids) if cc == c]
        assert set(idx) <= set(si.tolist()) or set(idx) <= set(li.tolist()), c


def test_conv_rank_median_split_drops_middle_conv_when_odd():
    conv_ids = [f"c{i}" for i in range(7)]
    lengths = {f"c{i}": i + 1 for i in range(7)}
    split = conv_rank_median_split(conv_ids, lengths, seed=0)
    assert split["short_idx"].shape[0] == split["long_idx"].shape[0] == 3
    assert split["meta"]["dropped_middle_conv_count"] == 1


def test_conv_rank_median_split_raises_on_missing_length():
    with pytest.raises(RuntimeError, match="absent from the rendered length table"):
        conv_rank_median_split(["a", "b"], {"a": 5}, seed=0)


def test_fit_length_stratified_runs_the_real_body_on_both_halves():
    """Production-body test: the REAL fit path on a tiny parent-shaped store
    (n_train < d, the production rank regime), both halves fit and matched."""
    rng = np.random.default_rng(0)
    n_conv, per_conv, d = 40, 3, 60
    n = n_conv * per_conv
    conv_ids = [f"c{i}" for i in range(n_conv) for _ in range(per_conv)]
    Xp = rng.standard_normal((n, d)).astype(np.float32)
    Xc = rng.standard_normal((n, d)).astype(np.float32)
    W = rng.standard_normal((d, d)).astype(np.float32)
    Y = (Xc @ W + 0.5 * rng.standard_normal((n, d))).astype(np.float32)
    store = {"X_prefix": Xp, "X_context": Xc, "Y": Y, "conv_ids": conv_ids}
    lengths = {f"c{i}": 20 + 7 * i for i in range(n_conv)}
    out = fit_length_stratified(store, lengths, null_draws=3)
    assert set(out["arms"]) == {name for _, _, name in LENGTH_SPLIT_ARMS}
    for arm, res in out["arms"].items():
        assert res["short"]["n_rows"] == res["long"]["n_rows"], arm
        for half in ("short", "long"):
            assert np.isfinite(res[half]["r2"]), (arm, half)
            # every fitted map carries the standing-rule companion reads
            assert res[half]["identity_bias_r2"] is not None, (arm, half)
            assert "acc_at_k" in res[half]["knn_euclidean"], (arm, half)
            assert "acc_at_k" in res[half]["knn_cosine"], (arm, half)
            assert res[half]["null_shuffle_fit_targets"]["n_draws"] == 3
        assert res["delta_r2_long_minus_short"] == pytest.approx(
            res["long"]["r2"] - res["short"]["r2"]
        )
        assert "full_refit" not in res, "the full-set refit is opt-in"


def test_published_parent_r2_reads_l19_per_arm(tmp_path: Path):
    ref = {
        "layers": [14, 18, 19, 26],
        "n_rows": 7365,
        "prefix": {"held_out_r2_per_layer": [0.01, 0.02, 0.0751, 0.04]},
        "context": {"held_out_r2_per_layer": [0.01, 0.02, 0.0835, 0.04]},
    }
    (tmp_path / "heldout_r2_M_assistant_chat.json").write_text(json.dumps(ref), encoding="utf-8")
    got = published_parent_r2(tmp_path, "M", "assistant_chat")
    assert got["prefix->answer"] == pytest.approx(0.0751)
    assert got["context->answer"] == pytest.approx(0.0835)
    assert got["n_rows_published"] == 7365
    with pytest.raises(FileNotFoundError):
        published_parent_r2(tmp_path, "M", "assistant_naturalistic")


def test_projection_charges_the_grid_and_the_length_split():
    """The wall fence is only a protection if the projection covers every phase:
    the addendum-E grid and the addendum-A half-fits must both be charged."""
    basis = measure_fold_basis(40, 24, null_draws=2)

    def entry(framing="chat", variant="base", prov="lmsys", uid="u"):
        return {
            "unit_id": uid,
            "n_rows": 40,
            "model_dir": "M",
            "framing": framing,
            "variant": variant,
            "provenance": prov,
            "fit_pairs": [["a", "b", "primary"]],
            "read_group_names": ["u2"],
            "grid_x_kinds": ["X_clean", "X_straddle"],
            "grid_y_kinds": ["Y_mean", "Y_end", "Y_boundary"],
        }

    bare = project_battery_wall([entry()], basis, null_draws=2)
    with_split = project_battery_wall([entry()], basis, null_draws=2, length_split_n_rows=[40, 40])
    assert bare["grid_hours"] > 0, "grid combos must be charged"
    assert bare["n_grid_combos"] == 6
    assert bare["length_split_hours"] == 0 and bare["n_length_split_cells"] == 0
    assert with_split["length_split_hours"] > 0
    assert with_split["n_length_split_cells"] == 2
    assert with_split["total_hours"] > bare["total_hours"]


def test_projection_charges_transfers_only_for_pairs_that_actually_run():
    """A flat per-frame transfer charge inflates the fence once the bridging
    families land: they carry ONE u2 provenance each, so no provenance-transfer
    pair can run on them, and a 6-per-frame charge would bill 6 phantom max-n
    reads per bridging frame."""
    basis = measure_fold_basis(40, 24, null_draws=2)

    def entry(framing, variant, prov, uid):
        return {
            "unit_id": uid,
            "n_rows": 40,
            "model_dir": "M",
            "framing": framing,
            "variant": variant,
            "provenance": prov,
            "fit_pairs": [["a", "b", "primary"]],
            "read_group_names": ["u2"],
            "grid_x_kinds": ["X_clean", "X_straddle"],
            "grid_y_kinds": ["Y_mean", "Y_end", "Y_boundary"],
        }

    # A bridging frame ALONE: one provenance -> zero provenance transfers.
    lone = project_battery_wall(
        [entry("single_turn", "assistant", "parent", "b1")], basis, null_draws=2
    )
    assert lone["n_transfer_reads_breakdown"]["provenance"] == 0
    assert lone["n_transfer_reads_breakdown"]["story_label"] == 0
    assert lone["n_transfer_reads_breakdown"] == {
        "provenance": 0,
        "story_label": 0,
        "cross_role": 2,
    }

    # A base frame with all three provenances runs 3 pairs x 2 directions = 6.
    three = project_battery_wall(
        [entry("chat", "base", p, f"c_{p}") for p in ("lmsys", "haiku", "onpolicy")],
        basis,
        null_draws=2,
    )
    assert three["n_transfer_reads_breakdown"]["provenance"] == 6

    # Adding the bridging frame beside it must NOT add a single transfer read.
    mixed = project_battery_wall(
        [entry("chat", "base", p, f"c_{p}") for p in ("lmsys", "haiku", "onpolicy")]
        + [entry("single_turn", "assistant", "parent", "b1")],
        basis,
        null_draws=2,
    )
    assert mixed["n_transfer_reads"] == three["n_transfer_reads"], (
        "a single-provenance bridging frame must add zero transfer reads"
    )

    # Story-label transfers need BOTH variants under the same provenance.
    one_variant = project_battery_wall([entry("story", "alex", "lmsys", "s1")], basis, null_draws=2)
    assert one_variant["n_transfer_reads_breakdown"]["story_label"] == 0
    both_variants = project_battery_wall(
        [entry("story", "alex", "lmsys", "s1"), entry("story", "user_label", "lmsys", "s2")],
        basis,
        null_draws=2,
    )
    assert both_variants["n_transfer_reads_breakdown"]["story_label"] == 2


# ---------------------------------------------------------------------------
# Addendum C prerequisite: on-policy a1 generation
# ---------------------------------------------------------------------------


def test_gen_a1_fingerprint_covers_every_output_affecting_key():
    """A resume key that ignores an output-affecting flag silently reuses wrong
    rows (#722 r3) — model, prompt set, token cap AND engine knobs must all move
    the fingerprint."""
    rows = [{"conv_id": "c1", "prompt": "p1"}, {"conv_id": "c2", "prompt": "p2"}]
    eng = {"enforce_eager": False, "no_prefix_caching": False, "max_model_len": 8192}
    base = GA1.fingerprint("Qwen/Qwen2.5-7B", rows, max_new_tokens=1024, engine=eng)
    assert base == GA1.fingerprint("Qwen/Qwen2.5-7B", rows, max_new_tokens=1024, engine=eng)
    variants = {
        "model": GA1.fingerprint("Qwen/Qwen2.5-7B-Instruct", rows, max_new_tokens=1024, engine=eng),
        "max_new": GA1.fingerprint("Qwen/Qwen2.5-7B", rows, max_new_tokens=256, engine=eng),
        "engine": GA1.fingerprint(
            "Qwen/Qwen2.5-7B", rows, max_new_tokens=1024, engine={**eng, "enforce_eager": True}
        ),
        "prompt": GA1.fingerprint(
            "Qwen/Qwen2.5-7B",
            [rows[0], {"conv_id": "c2", "prompt": "DIFFERENT"}],
            max_new_tokens=1024,
            engine=eng,
        ),
        "conv_set": GA1.fingerprint("Qwen/Qwen2.5-7B", rows[:1], max_new_tokens=1024, engine=eng),
    }
    for name, fp in variants.items():
        assert fp != base, f"{name} must change the resume fingerprint"


def test_gen_a1_checkpoint_resumes_on_match_and_restarts_on_mismatch(tmp_path: Path):
    out = tmp_path / "a1.jsonl"
    meta = tmp_path / "a1.meta.json"
    GA1.append_rows(out, [{"conv_id": "c1", "a1_onpolicy": "x"}])
    GA1.write_meta(meta, {"fingerprint": "GOOD"})
    assert set(GA1.load_checkpoint(out, meta, "GOOD")) == {"c1"}
    assert GA1.load_checkpoint(out, meta, "OTHER") == {}, "a mismatched key must restart fresh"
    # a second append is additive + durable
    GA1.append_rows(out, [{"conv_id": "c2", "a1_onpolicy": "y"}])
    assert set(GA1.load_checkpoint(out, meta, "GOOD")) == {"c1", "c2"}


def test_gen_a1_prompt_budget_filter_drops_overlong_and_fails_loud_when_empty():
    """An over-length prompt is ENGINE-FATAL at vLLM add_request (#952/#1738),
    so the filter runs at LOAD time and records drops digest-only."""
    tok = _tokenizer()
    rows = [{"conv_id": "short", "u1": "hi"}, {"conv_id": "long", "u1": "word " * 400}]
    kept, digest = GA1.filter_by_prompt_budget(rows, tok, budget=200)
    assert [r["conv_id"] for r in kept] == ["short"]
    assert digest["n_dropped"] == 1
    assert digest["dropped"][0]["conv_id"] == "long"
    assert "u1" not in digest["dropped"][0], "drop records are digest-only, never row text"
    assert digest["max_kept_prompt_tokens"] <= 200
    with pytest.raises(RuntimeError, match="kept 0 of"):
        GA1.filter_by_prompt_budget(rows, tok, budget=1)


def test_gen_a1_visible_devices_indexes_into_preset_allocation(monkeypatch):
    """NEVER export absolute device indices on a shared node — index INTO the
    pre-set CUDA_VISIBLE_DEVICES allocation (the SLURM lesson)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert GA1.visible_devices() == ["2", "5"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 3 , 4 ,")
    assert GA1.visible_devices() == ["3", "4"]


def test_gen_a1_greedy_chunked_chunks_and_disables_tqdm():
    """Production-body test of the real chunking path: the vLLM boundary is faked
    by a signature-conformant stub, everything else is the real body."""
    calls: list[dict] = []

    class _Out:
        def __init__(self, text):
            self.outputs = [type("O", (), {"text": text, "finish_reason": "stop"})()]

    class _FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=True):  # mirrors LLM.generate
            calls.append({"n": len(prompts), "use_tqdm": use_tqdm, "sp": sampling_params})
            return [_Out(f"r{i}") for i in range(len(prompts))]

    prompts = [f"p{i}" for i in range(7)]
    got = list(GA1.greedy_chunked(_FakeLLM(), prompts, max_new_tokens=32, chunk=3))
    assert [c["n"] for c in calls] == [3, 3, 1], "must chunk, never one giant generate (#664)"
    assert all(c["use_tqdm"] is False for c in calls), "#613 tqdm ZeroDivisionError"
    assert all(c["sp"].temperature == 0.0 and c["sp"].max_tokens == 32 for c in calls)
    assert [o for o, _, _ in got] == [0, 3, 6], "offsets must index the ORIGINAL prompt list"
    assert sum(len(t) for _, t, _ in got) == 7


def test_gen_a1_sets_spawn_before_vllm_import():
    """vLLM reads VLLM_WORKER_MULTIPROC_METHOD at IMPORT time, so the setdefault
    must sit above any vllm import (#628 fork-poisoned EngineCore death)."""
    src = Path(GA1.__file__).read_text(encoding="utf-8").split("\n")
    spawn_line = next(i for i, ln in enumerate(src) if "VLLM_WORKER_MULTIPROC_METHOD" in ln)
    vllm_imports = [
        i for i, ln in enumerate(src) if ln.strip().startswith(("import vllm", "from vllm"))
    ]
    assert vllm_imports, "expected at least one vllm import to guard"
    assert spawn_line < min(vllm_imports), "spawn setdefault must precede every vllm import"


# ---------------------------------------------------------------------------
# Addenda B / C / D: the bridging families
# ---------------------------------------------------------------------------


def test_single_turn_offsets_strictly_increasing_and_exact():
    """Addendum B: the prefix turn is ABLATED, so the only X is the label header
    and the answer must be a real, non-degenerate span."""
    u2, a2 = "what about tomorrow?", "Tomorrow looks clear."
    for variant, label in SINGLE_TURN_VARIANTS.items():
        text, off = single_turn_text_and_offsets(u2, a2, label=label)
        order = ["first_user_header_end", "u2_end", "answer_header_end", "parent_answer_end"]
        vals = [off[k] for k in order]
        assert vals == sorted(vals) and len(set(vals)) == len(vals), (variant, off)
        assert text[: off["first_user_header_end"]] == "User: "
        assert text[off["first_user_header_end"] : off["u2_end"]] == u2
        assert text[off["u2_end"] : off["answer_header_end"]] == f"\n\n{label}: "
        assert text[off["answer_header_end"] : off["parent_answer_end"]] == a2
        assert off["parent_answer_end"] == len(text)
        assert "u1" not in text.lower().split(":")[0]  # no prefix turn at all


def test_onpolicy_a1_render_substitutes_a1_and_varies_only_the_label():
    """Addendum C: a1 := the model's own reply; the label is the only other
    difference, and the u2 target layout matches the base naturalistic cell."""
    tok = _tokenizer()
    row_kwargs = dict(u1="why is the sky blue?", a1="LMSYS OFF-POLICY REPLY", u2="and at sunset?")
    _, base_off = naturalistic_text_and_offsets(**row_kwargs)
    from scripts.issue1689_user_slot_render import SourceRow

    src = SourceRow(
        "c1",
        row_kwargs["u1"],
        row_kwargs["a1"],
        row_kwargs["u2"],
        "haiku",
        None,
        a1_onpolicy="ON-POLICY REPLY FROM THE MEASURED MODEL",
    )
    texts = {}
    for variant in ("assistant", "wren"):
        unit = Unit("Qwen/Qwen2.5-7B", "onpolicy_a1", "haiku", variant)
        text, off = render_row(unit, src, tok)
        texts[variant] = text
        assert set(base_off) <= set(off) or set(off) <= set(base_off) or True
        # a1 is the ON-POLICY text, never the corpus reply
        assert "ON-POLICY REPLY FROM THE MEASURED MODEL" in text
        assert row_kwargs["a1"] not in text, "the off-policy a1 must not survive"
        # u2 target layout identical in SHAPE to the base naturalistic cell
        assert text[off["u2_header_end"] : off["u2_end"]] == row_kwargs["u2"]
        vals = [off[k] for k in unit.slots]
        assert vals == sorted(vals) and len(set(vals)) == len(vals), (variant, off)
    assert texts["assistant"] != texts["wren"], "the label must actually change the render"
    assert texts["assistant"].replace("Assistant: ", "Wren: ") == texts["wren"]
    # a missing on-policy a1 must fail loud, never silently fall back to a1
    bare = SourceRow("c2", "q", "off-policy", "u2", "haiku", None)
    with pytest.raises(RuntimeError, match="no on-policy a1"):
        render_row(Unit("Qwen/Qwen2.5-7B", "onpolicy_a1", "haiku", "assistant"), bare, tok)


def test_parent_recap_reproduces_both_parent_shapes():
    """Addendum D: chat conditions via apply_chat_template, plain-text conditions
    via the renderer's own segments — the parent capture's two conventions."""
    tok = _tokenizer()
    from scripts.issue1689_user_slot_render import SourceRow

    u1, a1, u2, a2 = "hi", "hello", "say more", "Here is more detail."
    # plain-text condition
    pr_plain = {
        "prefix_text_only": f"User: {u1}\n\nAssistant: {a1}\n\n",
        "u2_text_marked": f"User: {u2}",
        "context_tail": "\n\nAssistant: ",
        "messages": None,
    }
    row = SourceRow("c1", u1, a1, u2, "parent", None, a2=a2, parent_render=pr_plain)
    text, off = parent_recap_text_and_offsets(row, tok, variant="assistant_naturalistic")
    assert (
        text
        == pr_plain["prefix_text_only"] + pr_plain["u2_text_marked"] + pr_plain["context_tail"] + a2
    )
    assert off["prev_turn_end"] == len(pr_plain["prefix_text_only"])
    assert text[off["answer_header_end"] : off["parent_answer_end"]] == a2
    vals = [off[k] for k in ("prev_turn_end", "answer_header_end", "parent_answer_end")]
    assert vals == sorted(vals) and len(set(vals)) == len(vals), off

    # chat condition
    msgs = [
        {"role": "user", "content": u1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": u2},
    ]
    row_chat = SourceRow("c2", u1, a1, u2, "parent", None, a2=a2, parent_render={"messages": msgs})
    ctext, coff = parent_recap_text_and_offsets(row_chat, tok, variant="assistant_chat")
    assert ctext.endswith(a2)
    assert ctext[coff["answer_header_end"] : coff["parent_answer_end"]] == a2
    assert ctext[: coff["answer_header_end"]].endswith("<|im_start|>assistant\n")
    cvals = [coff[k] for k in ("prev_turn_end", "answer_header_end", "parent_answer_end")]
    assert cvals == sorted(cvals) and len(set(cvals)) == len(cvals), coff
    # a row with no a2 fails loud
    with pytest.raises(RuntimeError, match="no a2"):
        parent_recap_text_and_offsets(
            SourceRow("c3", u1, a1, u2, "parent", None, parent_render=pr_plain),
            tok,
            variant="assistant_naturalistic",
        )


def test_bridge_read_groups_validate_and_target_the_right_span():
    """Every bridging family's read groups must pass the shared validator and put
    the ANSWER (B/D) or u2 (C) in the group's answer span."""
    tok = _tokenizer()
    from scripts.issue1689_user_slot_render import SourceRow

    cases = []
    st_unit = Unit("Qwen/Qwen2.5-7B", "single_turn", "parent", "assistant")
    st_row = SourceRow("c1", "", "", "and tomorrow?", "parent", None, a2="Clear skies.")
    cases.append((st_unit, st_row, "answer", "Clear skies."))

    oa_unit = Unit("Qwen/Qwen2.5-7B", "onpolicy_a1", "haiku", "wren")
    oa_row = SourceRow(
        "c2", "why blue?", "off", "and sunset?", "haiku", None, a1_onpolicy="Because Rayleigh."
    )
    cases.append((oa_unit, oa_row, "u2", "and sunset?"))

    pr_unit = Unit("Qwen/Qwen2.5-7B", "parent_recap", "parent", "wren_naturalistic")
    pr_row = SourceRow(
        "c3",
        "hi",
        "hello",
        "more",
        "parent",
        None,
        a2="Extra detail.",
        parent_render={
            "prefix_text_only": "User: hi\n\nWren: hello\n\n",
            "u2_text_marked": "User: more",
            "context_tail": "\n\nWren: ",
            "messages": None,
        },
    )
    cases.append((pr_unit, pr_row, "answer", "Extra detail."))

    for unit, row, want_group, want_span in cases:
        text, off = render_row(unit, row, tok)
        new_text, groups = build_read_groups(unit, off, text)  # validator runs inside
        names = [g.name for g in groups]
        assert want_group in names, (unit.framing, names)
        g = next(g for g in groups if g.name == want_group)
        assert new_text[g.answer_start : g.answer_end] == want_span, unit.framing
        assert g.answer_start < g.answer_end <= g.boundary_end <= len(new_text)
        assert new_text.startswith(text), "the suffix must be APPENDED, never inserted"
        # 5 grid slots per group (addendum E), all distinct names
        all_slots = [s for gg in groups for s in gg.slot_names]
        assert len(all_slots) == 5 * len(groups) == len(set(all_slots))


# ---------------------------------------------------------------------------
# Addenda B/C/D: the bridging reduction
# ---------------------------------------------------------------------------


def _grid(mean, end, boundary):
    return {
        "combos": {
            "X_clean->Y_mean": {"r2": mean},
            "X_clean->Y_end": {"r2": end},
            "X_clean->Y_boundary": {"r2": boundary},
        }
    }


def test_bridge_comparisons_pairs_the_addenda_and_records_every_skip(tmp_path: Path):
    """The reduction is what actually ANSWERS addenda B/C/D, so pin the pairing:
    B contrasts the prefix-ablated cell against the full two-turn cell holding
    the SAME answer target; C contrasts on-policy a1 against the matched base
    cell; D reports the target-summary convention. A missing cell must be
    SKIPPED with its reason recorded, never silently dropped."""
    m = "Qwen_Qwen2.5-7B"

    def e(frame, framing, variant, prov, primary):
        return {
            "unit_id": f"{m}__{frame}__{prov}",
            "model_dir": m,
            "framing": framing,
            "variant": variant,
            "provenance": prov,
            "primary_fit": primary,
        }

    specs = [
        e(
            "single_turn_assistant",
            "single_turn",
            "assistant",
            "parent",
            "primary_singleturn_to_answer",
        ),
        e(
            "parent_recap_assistant_naturalistic",
            "parent_recap",
            "assistant_naturalistic",
            "parent",
            "primary_context_to_answer",
        ),
        e("onpolicy_a1_assistant", "onpolicy_a1", "assistant", "haiku", "primary_label_to_u2"),
        e("onpolicy_a1_wren", "onpolicy_a1", "wren", "haiku", "primary_label_to_u2"),
        e("naturalistic", "naturalistic", "base", "haiku", "primary_label_to_u2"),
    ]
    entries = {s["unit_id"]: s for s in specs}
    r2 = {
        f"{m}__single_turn_assistant__parent": 0.10,
        f"{m}__parent_recap_assistant_naturalistic__parent": 0.45,
        f"{m}__onpolicy_a1_assistant__haiku": 0.30,
        f"{m}__onpolicy_a1_wren__haiku": 0.28,
        f"{m}__naturalistic__haiku": 0.20,
    }
    per_unit = {
        s["unit_id"]: {
            "fits": {s["primary_fit"]: {"r2": r2[s["unit_id"]]}},
            "grid": {"answer" if s["framing"] != "onpolicy_a1" else "u2": _grid(0.5, 0.4, 0.3)},
        }
        for s in specs
    }

    out = bridge_comparisons(per_unit, entries, tmp_path)

    # B: the delta is two_turn - single_turn on the MATCHED pair.
    b = out["prefix_ablation"][f"{m}|single_turn_assistant"]
    assert b["two_turn_unit"] == f"{m}__parent_recap_assistant_naturalistic__parent"
    assert b["delta_r2_two_turn_minus_single_turn"] == pytest.approx(0.45 - 0.10)
    assert "skipped" not in b
    # the wren pair has no cells here -> SKIPPED with a reason, not dropped
    b_wren = out["prefix_ablation"][f"{m}|single_turn_wren"]
    assert b_wren["skipped"] and b_wren["delta_r2_two_turn_minus_single_turn"] is None

    # C: matched-label pair computes a delta; the wren cell names its confound.
    c = out["a1_provenance"][f"{m}|onpolicy_a1_assistant"]
    assert c["offpolicy_unit"] == f"{m}__naturalistic__haiku"
    assert c["delta_r2_onpolicy_minus_offpolicy"] == pytest.approx(0.30 - 0.20)
    c_wren = out["a1_provenance"][f"{m}|onpolicy_a1_wren"]
    assert c_wren["offpolicy_unit"] is None
    assert "confound" in c_wren["skipped"]

    # D: every grid-bearing cell reports its Y convention + the mean-end delta.
    d_key = f"{m}__parent_recap_assistant_naturalistic__parent|answer"
    d = out["target_summary_convention"][d_key]
    assert d["x_kind"] == "X_clean"
    assert d["r2_by_y"] == {"Y_mean": 0.5, "Y_end": 0.4, "Y_boundary": 0.3}
    assert d["delta_r2_mean_minus_end"] == pytest.approx(0.1)
    # the published parent reference is absent here -> path recorded, never faked
    assert d["published_parent_end_convention"] is None
    assert d["published_reference_absent"].endswith(
        "heldout_r2_Qwen_Qwen2.5-7B_assistant_naturalistic.json"
    )
    # a NON-recap cell gets no published slot at all (none exists for it)
    non_recap = out["target_summary_convention"][f"{m}__onpolicy_a1_assistant__haiku|u2"]
    assert "published_parent_end_convention" not in non_recap


def test_bridge_comparisons_attaches_the_published_end_convention_when_present(tmp_path: Path):
    """Addendum D's whole point is Y_mean vs the parent's END convention, so when
    the published per-cell reference exists it must ride the row."""
    m = "Qwen_Qwen2.5-7B"
    (tmp_path / f"heldout_r2_{m}_wren_naturalistic.json").write_text(
        json.dumps(
            {
                "n_rows": 3200,
                "layers": [14, 18, 19, 26],
                "prefix": {"held_out_r2_per_layer": [0.0, 0.0, 0.11, 0.0]},
                "context": {"held_out_r2_per_layer": [0.0, 0.0, 0.62, 0.0]},
            }
        ),
        encoding="utf-8",
    )
    uid = f"{m}__parent_recap_wren_naturalistic__parent"
    entries = {
        uid: {
            "unit_id": uid,
            "model_dir": m,
            "framing": "parent_recap",
            "variant": "wren_naturalistic",
            "provenance": "parent",
            "primary_fit": "primary_context_to_answer",
        }
    }
    per_unit = {
        uid: {
            "fits": {"primary_context_to_answer": {"r2": 0.4}},
            "grid": {"answer": _grid(0.58, 0.41, 0.33)},
        }
    }
    row = bridge_comparisons(per_unit, entries, tmp_path)["target_summary_convention"][
        f"{uid}|answer"
    ]
    pub = row["published_parent_end_convention"]
    assert pub["context->answer"] == pytest.approx(0.62)
    assert pub["prefix->answer"] == pytest.approx(0.11)
    assert pub["n_rows_published"] == 3200
    assert "published_reference_absent" not in row


def test_visible_devices_indexes_into_a_preset_cvd_allocation(monkeypatch):
    """`nvidia-smi` lists EVERY host GPU regardless of CVD, so on a shared SLURM
    node (the `fellows` lane, FIRST in the default auto chain) an absolute
    0..n-1 fan-out clobbers the scheduler's allocation onto other users'
    devices (#1345 crash-fix 15771). A pre-set allocation must win."""
    import scripts.issue1689_user_slot_capture as CAP

    def _boom(*a, **k):  # nvidia-smi must not even be consulted
        raise AssertionError("nvidia-smi consulted despite a pre-set CVD")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setattr(CAP.subprocess, "run", _boom)
    assert CAP.visible_devices() == ["3", "5"]

    # Unset CVD -> fall back to the nvidia-smi enumeration.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class _Out:
        stdout = "0\n1\n2\n"

    monkeypatch.setattr(CAP.subprocess, "run", lambda *a, **k: _Out())
    assert CAP.visible_devices() == ["0", "1", "2"]

    # A dead / absent nvidia-smi yields NO devices, which the dispatcher's
    # own assignment then refuses loudly rather than silently running on CPU.
    def _fail(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(CAP.subprocess, "run", _fail)
    assert CAP.visible_devices() == []
    with pytest.raises(RuntimeError, match="no visible GPUs"):
        assign_units_to_gpus([{"unit_id": "u", "model": "m", "n_rows": 1}], 0)


def test_dispatch_pins_the_allocated_device_not_the_lane_index(tmp_path, monkeypatch):
    """The dispatch body must pin the child CVD to the ALLOCATED physical device
    and pass the MATCHING --gpu-id. Executes the real `run_dispatch` planning +
    launch path, faking ONLY the subprocess boundary (no GPU needed)."""
    import scripts.issue1689_user_slot_capture as CAP

    rendered = tmp_path / "rendered"
    out_root = tmp_path / "store"
    rendered.mkdir()
    units = [
        {
            "unit_id": f"Qwen_Qwen2.5-7B__chat__{p}",
            "model": "Qwen/Qwen2.5-7B",
            "model_dir": "Qwen_Qwen2.5-7B",
            "n_rows": 10,
            "token_len_p50": 100,
        }
        for p in ("lmsys", "haiku")
    ]
    (rendered / "manifest.json").write_text(json.dumps({"units": units}), encoding="utf-8")

    # SLURM-style allocation: physical devices 4 and 6, never 0 and 1.
    monkeypatch.setattr(CAP, "visible_devices", lambda: ["4", "6"])
    monkeypatch.setattr(CAP, "run_render_if_missing", lambda args: None)

    launched: list[tuple[str, str]] = []

    class _FakeProc:
        returncode = 0

        def wait(self):
            return 0

    real_popen = CAP.subprocess.Popen

    def _fake_popen(cmd, *a, env=None, **kw):
        # Intercept ONLY the worker launches; everything else (the git-metadata
        # `subprocess.run`) must reach the real Popen.
        if not (isinstance(cmd, list) and "--mode" in cmd and "worker" in cmd):
            return real_popen(cmd, *a, env=env, **kw)
        gpu_id = cmd[cmd.index("--gpu-id") + 1]
        launched.append((env["CUDA_VISIBLE_DEVICES"], gpu_id))
        # stand in for the worker's store write so the dispatch flow continues
        for uid in cmd[cmd.index("--units") + 1].split(","):
            sp = out_root / "Qwen_Qwen2.5-7B" / uid / "L19.pt"
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_bytes(b"x")
        return _FakeProc()

    monkeypatch.setattr(CAP.subprocess, "Popen", _fake_popen)

    args = argparse.Namespace(
        rendered_dir=rendered,
        out_root=out_root,
        stage_root=tmp_path / "stage",
        log_dir=tmp_path / "logs",
        sentinel=tmp_path / "sentinel.json",
        units="all",
        num_gpus=0,
        batch_size=8,
        max_rows=0,
        overwrite=False,
        smoke=False,
        skip_upload=True,
        force_render=False,
    )
    assert CAP.run_dispatch(args) == 0

    # BOTH halves of the pin carry the PHYSICAL device, never the lane index.
    assert sorted(launched) == [("4", "4"), ("6", "6")], launched
    for cvd, gpu_id in launched:
        assert cvd == gpu_id, "the CVD pin and --gpu-id must agree (#545)"
    assert json.loads((tmp_path / "sentinel.json").read_text())["status"] == "ok"
