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
