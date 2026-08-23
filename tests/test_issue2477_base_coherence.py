"""CPU-only unit tests for the #2477 base-coherence driver (round-2 fixes).

No model, no GPU, no network. Covers: the inventory classifier pipeline
(``_classify_file``/``_classify_bank`` — exclusion families, ROOT_FAMILY
family-exclusion instead of the round-1 ``/base/``-token Qwen fallback,
indeterminate fail-direction, the synthetic positive arm), the A5 bank-parity
gate's failure branches (with a signature-conformant ``_stage_file`` fake at
the network boundary; the vacuous-sidecar recipe-parity feed is documented as
the OPEN ``bank-parity-contingency-broken`` concern — its safe direction is
gate-FAIL => generation fires), the gen manifest-contract assert (each failure
class), the judge-wave completeness predicate, the cross-shard merge guard,
and the fresh-rows panel contract.

Content hygiene: all fixture text is synthetic placeholder prose — no real
corpus rows appear anywhere in this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2477_base_coherence as mod  # noqa: E402

# ---------------------------------------------------------------------------
# _classify_file / _classify_bank
# ---------------------------------------------------------------------------


def _cls(path: str, payloads: dict | None = None, size: int = 100) -> dict:
    return mod._classify_file(path, size, payloads or {})


def test_classify_analysis_tensors_excluded():
    row = _cls("issue2061_sae_predictability/analysis_tensors/row_index.json")
    assert row["classification"].startswith("excluded:analysis-tensors")
    assert row["model"] is None


def test_classify_allowlist_and_audit_excluded():
    for name in ("allowlist.json", "audit.json"):
        row = _cls(f"issue1336_rlvr_ladder/raw_completions/generation/base/{name}")
        assert row["classification"].startswith("excluded:filter-audit-artifact"), row


def test_classify_corpus_and_steer_probe_excluded():
    row = _cls("issue1902_stage_map/corpus/prompts.jsonl")
    assert row["classification"].startswith("excluded:corpus-input")
    row = _cls("issue1902_stage_map/steer_probe/probe_summary.json")
    assert row["classification"].startswith("excluded:steer-probe-artifact")


def test_classify_non_text_format_excluded():
    row = _cls("issue825_userbase_map/raw_completions/tensors.npz")
    assert row["classification"].startswith("excluded:non-text-format")


def test_classify_meta_sidecar():
    row = _cls("issue825_userbase_map/raw_completions/track_s/track_s_meta.json")
    assert row["classification"] == "meta-sidecar"


def test_classify_1336_base_rung_is_family_excluded_not_qwen():
    """Round-1 blocker: generation/base/ is the Llama/Tulu LADDER RUNG named 'base',
    never a Qwen bank — the bare `/base/` token rule is gone."""
    row = _cls("issue1336_rlvr_ladder/raw_completions/generation/base/answers.jsonl")
    assert row["classification"] == "completion-bank"
    assert row["absence_decision"] == "family-excluded"
    assert "Llama/Tulu" in row["model"]
    assert "qwen2.5-7b" not in row["model"].lower().replace(" ", "")[:20]
    assert row["is_base_generated_chat_template_bank"] is False


def test_classify_1336_naturalistic_twin_render():
    row = _cls(
        "issue1336_rlvr_ladder/raw_completions/generation/base__gen_naturalistic/answers.jsonl"
    )
    assert row["absence_decision"] == "family-excluded"
    assert row["render"].startswith("plain User:/Assistant: re-render")


def test_classify_1902_family_excluded():
    row = _cls("issue1902_stage_map/gen/step100.jsonl")
    assert row["absence_decision"] == "family-excluded"
    assert "OLMo-2" in row["model"]


def test_classify_track_s_determinate_instruct_chat():
    payloads = {
        "track_s_meta.json": {
            "model_instruct": "Qwen/Qwen2.5-7B-Instruct",
            "sampling": {"temperature": 1.0},
        }
    }
    row = _cls("issue825_userbase_map/raw_completions/track_s/track_s.jsonl", payloads)
    assert row["absence_decision"] == "determinate"
    assert row["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert row["render"] == "chat template"
    assert row["is_base_generated_chat_template_bank"] is False  # instruct, not base


def test_classify_armg_pretrained_determinate_plain_render():
    row = _cls(
        "issue825_userbase_map/raw_completions/turn_dynamics/armG/pretrained/shard0of3/step2.jsonl"
    )
    assert row["absence_decision"] == "determinate"
    assert row["model"] == mod.MODEL_BASE
    # base model but PLAIN render => not the A5 bank
    assert row["is_base_generated_chat_template_bank"] is False


def test_classify_unknown_bank_is_indeterminate_never_defaulted():
    row = _cls("issue825_userbase_map/raw_completions/mystery/bank.jsonl")
    assert row["absence_decision"] == "indeterminate"
    assert row["model"] == "undetermined"
    assert row["is_base_generated_chat_template_bank"] is False


def test_classify_synthetic_qwen_base_chat_sidecar_is_positive():
    """The positive arm: a sidecar-resolved Qwen base + chat-template bank IS flagged."""
    payloads = {
        "gen_meta.json": {
            "model": "Qwen/Qwen2.5-7B",
            "render": "apply_chat_template(add_generation_prompt=True)",
        }
    }
    row = _cls("issue825_userbase_map/raw_completions/somewhere/bank.jsonl", payloads)
    assert row["absence_decision"] == "determinate"
    assert row["is_base_generated_chat_template_bank"] is True


# ---------------------------------------------------------------------------
# _bank_parity_gate (network boundary faked with a signature-conformant fake)
# ---------------------------------------------------------------------------


def test_bank_parity_gate_no_candidates():
    gate = mod._bank_parity_gate([], [{"prompt_idx": 0, "prompt": "q0"}])
    assert gate["skip_generation"] is False
    assert gate["checks"] == []


def test_bank_parity_gate_panel_mismatch_fails_and_generation_fires(tmp_path, monkeypatch):
    bank = tmp_path / "candidate.jsonl"
    mod._write_jsonl(bank, [{"prompt_idx": 999, "prompt": "other"}])

    def _fake_stage_file(repo_path: str, min_free_note: str = "") -> Path:
        """Signature mirror of mod._stage_file; serves the local fixture, no network."""
        return bank

    monkeypatch.setattr(mod, "_stage_file", _fake_stage_file)
    chat_items = [{"prompt_idx": 0, "prompt": "q0"}, {"prompt_idx": 1, "prompt": "q1"}]
    gate = mod._bank_parity_gate(["some/candidate.jsonl"], chat_items)
    assert gate["skip_generation"] is False
    rec = gate["checks"][0]
    assert rec["passed"] is False
    assert any("prompt-panel identity" in f for f in rec["failures"])
    # OPEN concern bank-parity-contingency-broken (documented, safe direction): the
    # sidecar feed `[(cand, 0)]` is vacuous — recipe parity cannot pass, so a candidate
    # can only FAIL the gate and Phase C fires anyway.
    assert any("recipe parity" in f for f in rec["failures"])


# ---------------------------------------------------------------------------
# _assert_gen_manifest_contract
# ---------------------------------------------------------------------------


def _manifest(n: int = mod.N_CHAT) -> dict:
    return {
        "chat_items": [{"prompt_idx": i, "prompt": f"synthetic prompt {i}"} for i in range(n)],
        "meta": {
            "sampling_recipe": {
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 1024,
                "seed": mod.GEN_SEED,
            }
        },
    }


def test_gen_manifest_contract_passes_production_and_smoke():
    m = _manifest()
    mod._assert_gen_manifest_contract(m, m["chat_items"], smoke=False)
    mod._assert_gen_manifest_contract(m, m["chat_items"][:5], smoke=True)


@pytest.mark.parametrize(
    ("mutate", "want"),
    [
        (lambda m, c: (m["chat_items"].pop(), None), "manifest chat_items count"),
        (lambda m, c: (c.pop(), None), "post-slice chat_items count"),
        (lambda m, c: c[0].__setitem__("prompt_idx", True), "not a plain int"),
        (lambda m, c: c[1].__setitem__("prompt_idx", c[0]["prompt_idx"]), "not unique"),
        (lambda m, c: c[0].__setitem__("prompt", "  "), "empty/non-str prompt"),
        (
            lambda m, c: m["meta"]["sampling_recipe"].__setitem__("temperature", 0.7),
            "sampling_recipe[temperature]",
        ),
    ],
)
def test_gen_manifest_contract_failure_classes(mutate, want):
    m = _manifest()
    chat_items = m["chat_items"][:5]
    mutate(m, chat_items)
    with pytest.raises(RuntimeError, match="gen manifest contract failed") as ei:
        mod._assert_gen_manifest_contract(m, chat_items, smoke=True)
    assert want in str(ei.value)


# ---------------------------------------------------------------------------
# _judge_wave_complete
# ---------------------------------------------------------------------------


def _write_judge_raw(judge_dir: Path, arm: str, iids: list[str]) -> None:
    payload = {
        "all_scores": {f"{iid}__{0:05d}__{d:02d}": {"score": 80} for iid in iids for d in (0, 1)}
    }
    (judge_dir / f"judge_raw_{arm}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_judge_wave_complete_true_on_exact_coverage(tmp_path):
    expected = {"arm_a": {"arm_a--1", "arm_a--2"}, "arm_b": {"arm_b--x--d2"}}
    for arm, iids in expected.items():
        _write_judge_raw(tmp_path, arm, sorted(iids))
    assert mod._judge_wave_complete(tmp_path, expected) is True


def test_judge_wave_complete_false_on_missing_partial_or_malformed(tmp_path):
    expected = {"arm_a": {"arm_a--1", "arm_a--2"}, "arm_b": {"arm_b--x--d2"}}
    # missing arm_b file
    _write_judge_raw(tmp_path, "arm_a", ["arm_a--1", "arm_a--2"])
    assert mod._judge_wave_complete(tmp_path, expected) is False
    # partial coverage on arm_b
    _write_judge_raw(tmp_path, "arm_b", [])
    assert mod._judge_wave_complete(tmp_path, expected) is False
    # malformed JSON
    (tmp_path / "judge_raw_arm_b.json").write_text("{not json", encoding="utf-8")
    assert mod._judge_wave_complete(tmp_path, expected) is False


# ---------------------------------------------------------------------------
# _merge_shard_rows / _validate_fresh_rows_contract
# ---------------------------------------------------------------------------


def test_merge_shard_rows_disjoint_merges_and_overlap_raises():
    dst = {("c1", 2): {"answer": "a"}}
    mod._merge_shard_rows(dst, {("c2", 2): {"answer": "b"}}, "pretrained", "shard1of3")
    assert set(dst) == {("c1", 2), ("c2", 2)}
    with pytest.raises(RuntimeError, match="cross-shard duplicate"):
        mod._merge_shard_rows(dst, {("c1", 2): {"answer": "z"}}, "pretrained", "shard2of3")


def test_validate_fresh_rows_contract():
    manifest = {
        "chat_items": [{"prompt_idx": 0, "prompt": "q0"}, {"prompt_idx": 1, "prompt": "q1"}]
    }
    rows = [{"prompt_idx": 0}, {"prompt_idx": 1}]
    mod._validate_fresh_rows_contract(rows, manifest, "base_chat_seed42.jsonl")
    with pytest.raises(RuntimeError, match="row count"):
        mod._validate_fresh_rows_contract(rows[:1], manifest, "base_chat_seed42.jsonl")
    with pytest.raises(RuntimeError, match="prompt_idx set mismatch"):
        mod._validate_fresh_rows_contract(
            [{"prompt_idx": 0}, {"prompt_idx": 7}], manifest, "base_chat_seed42.jsonl"
        )


# ---------------------------------------------------------------------------
# C0 condition-set registry + decoding-sensitivity paths (plan v5)
# ---------------------------------------------------------------------------


def test_condition_sets_registry_shape_and_parent_literals():
    assert set(mod.CONDITION_SETS) == {"parent", "decoding-sensitivity"}
    parent = mod.CONDITION_SETS["parent"]
    # The parent set reproduces the parent round's phase_gen literals VERBATIM (parity bar).
    assert parent == (
        ("base_chat", "chat", 1.0, ("<|im_end|>",)),
        ("base_bare", "bare", 1.0, ("\nUser:", "\n\nUser:")),
    )
    dec = mod.CONDITION_SETS["decoding-sensitivity"]
    assert [r[0] for r in dec] == [
        "base_chat_t07",
        "base_chat_t00",
        "base_bare_t07",
        "base_bare_t00",
    ]
    assert [r[2] for r in dec] == [0.7, 0.0, 0.7, 0.0]
    for _slug, render, _temp, stop in dec:
        ref = parent[0] if render == "chat" else parent[1]
        assert stop == ref[3]  # stop strings inherit the parent's per-render literals
    assert mod.DECSENS_ARM_NAMES == (
        "arm_base_chat_t07",
        "arm_base_chat_t00",
        "arm_base_bare_t07",
        "arm_base_bare_t00",
    )
    assert mod.DECSENS_PILOT_TARGET_TOTAL_DRAWS == 208
    assert [p[:2] for p in mod.DECSENS_PAIR_SPECS] == [
        ("chat_t07_minus_chat_t10", "arm_base_chat_t07"),
        ("chat_t00_minus_chat_t10", "arm_base_chat_t00"),
        ("bare_t07_minus_bare_t10", "arm_base_bare_t07"),
        ("bare_t00_minus_bare_t10", "arm_base_bare_t00"),
    ]


def test_parse_args_default_condition_set_is_parent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["issue2477_base_coherence.py", "--phase", "aggregate"])
    args = mod._parse_args()
    assert args.condition_set == "parent"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue2477_base_coherence.py",
            "--phase",
            "gen",
            "--condition-set",
            "decoding-sensitivity",
        ],
    )
    assert mod._parse_args().condition_set == "decoding-sensitivity"


def test_gen_manifest_contract_rejects_bad_registry_temperature(monkeypatch):
    m = _manifest()
    bad = {
        "parent": mod.CONDITION_SETS["parent"],
        "decoding-sensitivity": (("base_chat_tbad", "chat", 2.5, ("<|im_end|>",)),),
    }
    monkeypatch.setattr(mod, "CONDITION_SETS", bad)
    with pytest.raises(RuntimeError, match="gen manifest contract failed") as ei:
        mod._assert_gen_manifest_contract(m, m["chat_items"][:5], smoke=True)
    assert "temperature" in str(ei.value) and "base_chat_tbad" in str(ei.value)
    # bool must not alias a valid temperature (type gate: True would pass 0.0 <= t <= 2.0)
    monkeypatch.setattr(
        mod, "CONDITION_SETS", {"decoding-sensitivity": (("x", "chat", True, ("<|im_end|>",)),)}
    )
    with pytest.raises(RuntimeError, match="gen manifest contract failed"):
        mod._assert_gen_manifest_contract(m, m["chat_items"][:5], smoke=True)


def test_gen_decsens_requires_set_specific_out_root():
    ns = argparse.Namespace(
        smoke=False,
        force=False,
        condition_set="decoding-sensitivity",
        out="/workspace/results/issue_2477",
    )
    with pytest.raises(SystemExit, match="set-specific --out"):
        mod.phase_gen(ns)


def test_build_arms_decoding_sensitivity(monkeypatch, tmp_path):
    m = _manifest()
    monkeypatch.setattr(mod, "DECSENS_EVAL_DIR", tmp_path)
    fresh = tmp_path / "fresh_completions"
    for slug, _r, _t, _s in mod.CONDITION_SETS["decoding-sensitivity"]:
        rows = [
            {
                "prompt_idx": it["prompt_idx"],
                "prompt": it["prompt"],
                "response": f"synthetic response {slug} {it['prompt_idx']}",
                "finish_reason": "stop",
            }
            for it in m["chat_items"]
        ]
        mod._write_jsonl(fresh / f"{slug}_seed42.jsonl", rows)
    arms = mod.build_arms(m, condition_set="decoding-sensitivity")
    assert set(arms) == set(mod.DECSENS_ARM_NAMES)
    for name, a in arms.items():
        assert len(a.items) == mod.N_CHAT
        assert all(iid.startswith(f"{name}--") for iid, _q, _ans in a.items)
        assert all("__" not in iid for iid, _q, _ans in a.items)
        assert a.cap_hit and not any(a.cap_hit.values())
        assert all(isinstance(a.pair_key[iid], int) for iid, _q, _ans in a.items)
    # prompt-hash mismatch fail-fast (panel asserts unchanged on the decsens path)
    bad_rows = mod._read_jsonl(fresh / "base_chat_t07_seed42.jsonl")
    bad_rows[0]["prompt"] = "tampered synthetic prompt"
    mod._write_jsonl(fresh / "base_chat_t07_seed42.jsonl", bad_rows)
    with pytest.raises(RuntimeError, match="prompt text mismatch"):
        mod.build_arms(m, condition_set="decoding-sensitivity")


def test_decsens_verdict_tokens():
    assert mod._decsens_verdict_token(0.85, 0.80) == "render-and-sampling"
    assert mod._decsens_verdict_token(0.80, 0.80) == "render-and-sampling"  # Δ ≥ 0 boundary
    assert mod._decsens_verdict_token(0.28, 0.80) == "render-driven"


def test_parent_item_means_real_committed_judge_raw():
    """Plan v5 §4 C0 unit check: the cross-temperature comparator path runs against the REAL
    committed parent judge_raw files (via the production loader — git-blob fallback covers
    sparse checkouts) and reproduces the committed parent verdict's frac_coherent values."""
    verdict = mod._read_committed_json("eval_results/issue_2477/coherence_verdict.json")
    for arm in ("arm_base_chat", "arm_base_bare"):
        means = mod._parent_item_means(arm)
        committed = verdict["arms"][arm]["frac_coherent"]
        assert len(means) == committed["n_kept"]
        assert all(isinstance(k, int) for k in means)
        assert all(0.0 <= v <= 100.0 for v in means.values())
        n_coh = sum(1 for v in means.values() if v >= mod.COHERENT_THRESHOLD)
        assert n_coh == committed["n_coherent"]
        assert n_coh / len(means) == pytest.approx(committed["value"])
        # item means match the committed per-item map exactly (kept-draw semantics parity)
        per_item = verdict["per_item_mean_scores"][arm]
        assert {int(k.rsplit("--", 1)[1]): v for k, v in per_item.items()} == pytest.approx(means)


def test_parent_item_means_fail_fast(monkeypatch):
    monkeypatch.setattr(mod, "_read_committed_json", lambda rel: {"all_scores": {}})
    with pytest.raises(RuntimeError, match="all_scores empty"):
        mod._parent_item_means("arm_base_chat")
    monkeypatch.setattr(
        mod,
        "_read_committed_json",
        lambda rel: {"all_scores": {"other_arm--3__00000__00": {"score": 50}}},
    )
    with pytest.raises(RuntimeError, match="unexpected item id"):
        mod._parent_item_means("arm_base_chat")
    # all draws dropped (REFUSAL) => zero kept draws, fail loud
    monkeypatch.setattr(
        mod,
        "_read_committed_json",
        lambda rel: {"all_scores": {"arm_base_chat--0__00000__00": {"score": "REFUSAL"}}},
    )
    with pytest.raises(RuntimeError, match="zero kept draws"):
        mod._parent_item_means("arm_base_chat")


def test_paired_delta_vs_parent_exclusion_semantics():
    a = mod.ArmData(name="arm_base_chat_t07")
    stats = {
        "n_items": 4,
        "kept_scores": {
            "arm_base_chat_t07--0": 80.0,
            "arm_base_chat_t07--1": 60.0,
            "arm_base_chat_t07--3": 40.0,
        },
    }
    for iid in stats["kept_scores"]:
        a.pair_key[iid] = int(iid.rsplit("--", 1)[1])
    parent_means = {0: 70.0, 1: 65.0, 2: 50.0}  # idx 2 unkept on the new side, 3 on parent's
    out = mod._paired_delta_vs_parent(stats, a, parent_means)
    assert out["n_pairs"] == 2
    assert out["n_excluded_pairs"] == 2
    assert out["per_pair_delta"] == {"0": 10.0, "1": -5.0}
    assert out["mean_delta"] == pytest.approx(2.5)


def test_aggregate_decsens_core_synthetic(tmp_path, monkeypatch):
    """Executes the real _aggregate_decsens_core body end-to-end on a tiny synthetic fixture
    (real _arm_stats over real judge_raw files); only the parent comparator loader — which has
    its own real-committed-file test above — is monkeypatched."""

    def _mk_arm(slug: str, n: int = 4) -> mod.ArmData:
        name = f"arm_{slug}"
        a = mod.ArmData(name=name)
        for i in range(n):
            iid = f"{name}--{i}"
            a.items.append((iid, f"synthetic q {i}", f"synthetic answer {i} words " * (i + 1)))
            a.pair_key[iid] = i
            a.cap_hit[iid] = False
        return a

    arms = {f"arm_{s}": _mk_arm(s) for s, _r, _t, _st in mod.CONDITION_SETS["decoding-sensitivity"]}
    save_raw = {}
    for name, arm in arms.items():
        score = 90 if name.endswith("t07") else 40
        all_scores = {
            f"{iid}__{j:05d}__00": {"score": score, "stop_reason": "end_turn"}
            for j, (iid, _q, _ans) in enumerate(arm.items)
        }
        p = tmp_path / f"judge_raw_{name}.json"
        p.write_text(json.dumps({"all_scores": all_scores}), encoding="utf-8")
        save_raw[name] = p
    parent = {0: 50.0, 1: 50.0, 2: 50.0}  # idx 3 unkept on the parent side => 1 excluded pair
    monkeypatch.setattr(mod, "_parent_item_means", lambda arm: dict(parent))

    payload = mod._aggregate_decsens_core(arms, save_raw, tmp_path / "verdict.json")
    assert (tmp_path / "verdict.json").exists()
    assert payload["condition_set"] == "decoding-sensitivity"
    assert payload["verdict"]["arm"] == "arm_base_chat_t07"
    assert payload["verdict"]["frac_coherent"] == 1.0  # all t07 item means = 90 >= 50
    assert payload["verdict"]["token"] == "render-and-sampling"
    assert payload["arms"]["arm_base_chat_t00"]["frac_coherent"]["value"] == 0.0
    pairs = payload["paired_deltas_vs_parent_t10"]
    assert set(pairs) == {k for k, _n, _p in mod.DECSENS_PAIR_SPECS}
    for key, _new, _par in mod.DECSENS_PAIR_SPECS:
        assert pairs[key]["n_pairs"] == 3
        assert pairs[key]["n_excluded_pairs"] == 1
    assert pairs["chat_t07_minus_chat_t10"]["mean_delta"] == pytest.approx(40.0)
    assert pairs["chat_t00_minus_chat_t10"]["mean_delta"] == pytest.approx(-10.0)
    comp = payload["parent_comparators"]
    assert set(comp) == {"arm_base_chat", "arm_base_bare"}
    assert comp["arm_base_chat"]["frac_coherent"]["value"] == 1.0  # 50.0 >= threshold 50
    # per-item maps stripped from arms, kept in the dedicated sections
    assert "kept_scores" not in payload["arms"]["arm_base_chat_t07"]
    assert set(payload["per_item_mean_scores"]) == set(mod.DECSENS_ARM_NAMES)


def test_aggregate_and_figures_refuse_decsens_smoke():
    for phase, token in ((mod.phase_aggregate, "aggregate"), (mod.phase_figures, "figures")):
        ns = argparse.Namespace(smoke=True, condition_set="decoding-sensitivity")
        with pytest.raises(SystemExit, match=f"{token} --smoke is parent-only"):
            phase(ns)
