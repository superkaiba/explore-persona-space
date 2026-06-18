"""Issue #657 — CPU-only smoke for the v6 behavior-appropriate extraction paths.

v6 PIVOT: behavior-direction recipes are now behavior-appropriate, NOT a single
generic recipe (which failed K2 for refusal at launch #3):
  - refusal -> Arditi diff-in-means (scripts/issue657_extract_diffmean_direction.py)
  - em      -> #521 on_policy_em trained-shift re-extraction (same extractor,
               --behavior em; single on-HF issue_519/em_seed42 adapter, config-pinned)
  - marker  -> the #623 affordance recipe (scripts/issue623_extract_sycophancy_vector.py);
               K2-fail -> #521 on_policy_marker shift fallback (--behavior marker).

These tests catch the CPU-testable crash classes WITHOUT a GPU:
  (i)   marker's #623 artifact-generation step (generate_artifacts());
  (ii)  the new diff-mean extractor's import + CLI contract + adapter-config pin;
  (iii) a missing ANTHROPIC_API_KEY.

Layers:
  - ``test_dispatcher_*`` / ``test_extractor_*`` — ALWAYS run, no API, no GPU.
  - ``test_generate_artifacts_live_marker`` — REAL Claude call for marker only
    (refusal/em no longer use generate_artifacts in production); runs only when
    ANTHROPIC_API_KEY is present (skips loud otherwise so the keyless sweep stays
    green). This is the layer the implementer runs locally with ``.env`` sourced.

All CPU-only — no GPU, no HF model load (the adapter-config pin reads a tiny
config.json, not the weights).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = REPO_ROOT / "scripts/issue623_extract_sycophancy_vector.py"
DIFFMEAN_SCRIPT = REPO_ROOT / "scripts/issue657_extract_diffmean_direction.py"
DISPATCHER = REPO_ROOT / "scripts/issue657_extract.sh"

# Marker keeps the #623 affordance recipe (the only production path still using
# generate_artifacts). refusal/em route through the diff-mean extractor in v6.
MARKER_BEHAVIOR = "marker"
# All three behaviors still have a canonical dispatcher trait description (marker
# uses it in production; refusal/em retain one for the --print smoke / fallback docs).
NEW_BEHAVIORS = ("refusal", "marker", "em")


def _load_extract_module():
    """Load scripts/issue623_extract_sycophancy_vector.py as a module (no GPU import yet)."""
    spec = importlib.util.spec_from_file_location("issue623_extract_657smoke", EXTRACT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dispatcher_trait_description(behavior: str) -> str:
    """Read the dispatcher's CANONICAL trait description for a behavior.

    Uses the ``--print-trait-description`` flag so the smoke tests the EXACT
    string production passes to ``--trait-description`` (no string drift).
    """
    proc = subprocess.run(
        ["bash", str(DISPATCHER), "--print-trait-description", behavior],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"dispatcher refused behavior {behavior!r}: {proc.stderr}"
    desc = proc.stdout.strip()
    assert desc, f"empty trait description for {behavior!r}"
    return desc


# ── ALWAYS-ON: dispatcher trait descriptions (no API, no GPU) ────────────────


@pytest.mark.parametrize("behavior", NEW_BEHAVIORS)
def test_dispatcher_trait_description_present(behavior):
    """Each NEW behavior has a non-empty, substantive trait description."""
    desc = _dispatcher_trait_description(behavior)
    # A real paper-recipe description is a full sentence, not a label.
    assert len(desc) >= 40, f"{behavior} description too short to be a real trait spec: {desc!r}"
    assert "the model" in desc.lower(), f"{behavior} description should describe model behavior"


def test_dispatcher_rejects_unknown_behavior():
    """An unknown behavior name exits non-zero (fail-loud, no empty description)."""
    proc = subprocess.run(
        ["bash", str(DISPATCHER), "--print-trait-description", "not_a_real_behavior"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0


def test_dispatcher_pairs_each_behavior_with_its_own_description():
    """The dispatcher pairs each NEW behavior with its OWN description, never the
    sycophancy default. main()'s recipe-mismatch guard raises at runtime when a
    non-default --trait-name carries the default sycophancy description (a silent
    "sycophancy artifacts under a refusal label" bug); this asserts the
    dispatcher never feeds that mismatch in the first place."""
    from explore_persona_space.experiments.persona_decomp_623 import (
        SYCOPHANCY_TRAIT_DESCRIPTION,
        SYCOPHANCY_TRAIT_NAME,
    )

    seen: dict[str, str] = {}
    for behavior in NEW_BEHAVIORS:
        assert behavior != SYCOPHANCY_TRAIT_NAME
        desc = _dispatcher_trait_description(behavior)
        assert desc != SYCOPHANCY_TRAIT_DESCRIPTION, (
            f"{behavior} reuses the sycophancy description — main()'s guard would raise"
        )
        seen[behavior] = desc
    # Each behavior's description is distinct (no copy-paste collision).
    assert len(set(seen.values())) == len(NEW_BEHAVIORS), f"duplicate descriptions: {seen}"


# ── ALWAYS-ON: the NEW diff-mean / shift extractor (import + CLI + pins) ─────


def _load_diffmean_module():
    """Load scripts/issue657_extract_diffmean_direction.py as a module (no GPU)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("i657_diffmean_smoke", DIFFMEAN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diffmean_extractor_imports_and_exposes_modes():
    """The NEW extractor imports cleanly (cross-script #623 import + lazy vLLM) and
    exposes both trained-shift modes with their distinct adapter pins."""
    mod = _load_diffmean_module()
    assert set(mod.SHIFT_ADAPTERS) == {"em", "marker"}
    # em -> issue_519/em_seed42 (dropout 0.05); marker -> issue_519/marker_seed42 (0.0).
    assert mod.SHIFT_ADAPTERS["em"]["subfolder"] == "issue_519/em_seed42"
    assert mod.SHIFT_ADAPTERS["marker"]["subfolder"] == "issue_519/marker_seed42"
    assert mod.SHIFT_ADAPTERS["em"]["expected_config"]["lora_dropout"] == 0.05
    assert mod.SHIFT_ADAPTERS["marker"]["expected_config"]["lora_dropout"] == 0.0
    # both pin r=8/alpha=16/use_rslora=true/base=Qwen-2.5-7B-Instruct
    for beh in ("em", "marker"):
        exp = mod.SHIFT_ADAPTERS[beh]["expected_config"]
        assert exp["r"] == 8 and exp["lora_alpha"] == 16 and exp["use_rslora"] is True
        assert exp["base_model_name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"
    # callable surface the dispatcher + plan §5 name
    for fn in (
        "extract_diffmean",
        "extract_em_shift",
        "assert_shift_adapter_config",
        "load_instruction_bank",
    ):
        assert callable(getattr(mod, fn)), f"missing {fn}"


def test_diffmean_extractor_cli_rejects_unknown_behavior():
    """The CLI accepts only refusal/em/marker (argparse choices); anything else exits."""
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(DIFFMEAN_SCRIPT),
            "--behavior",
            "nope",
            "--output-dir",
            "/tmp/x",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr.lower() or "nope" in proc.stderr.lower()


def test_adapter_config_pin_rejects_mismatch(tmp_path: Path):
    """assert_shift_adapter_config FAILS LOUD on a config mismatch (the §4 pin):
    e.g. the dropped em_turner_seed42 r=32/alpha=256 would NOT pass the em pin."""
    import json

    mod = _load_diffmean_module()
    # a WRONG (turner-like) config -> mismatch -> ValueError
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 32,
                "lora_alpha": 256,
                "lora_dropout": 0.05,
                "use_rslora": True,
                "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            }
        )
    )
    with pytest.raises(ValueError, match="adapter config mismatch"):
        mod.assert_shift_adapter_config(bad, "em")

    # the CORRECT em pin passes
    good = tmp_path / "good"
    good.mkdir()
    (good / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "use_rslora": True,
                "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            }
        )
    )
    cfg = mod.assert_shift_adapter_config(good, "em")
    assert cfg["r"] == 8

    # a missing config file fails loud (fail-fast)
    with pytest.raises(FileNotFoundError):
        mod.assert_shift_adapter_config(tmp_path / "absent", "em")


def test_decoder_layers_resolves_base_and_peft():
    """_decoder_layers resolves the decoder ModuleList for BOTH a base HF model AND
    a PeftModel (the v6 EM-shift smoke crashed on `PeftModel.model.layers` ->
    AttributeError: 'Qwen2ForCausalLM' has no attribute 'layers'). A tiny 0.5B model
    keeps this CPU-fast; the resolver is structural, not model-size-dependent."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    mod = _load_diffmean_module()
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
    base_layers = mod._decoder_layers(base)
    assert len(base_layers) > 0
    # PEFT-wrap and confirm the resolver still finds the SAME number of decoder layers
    peft = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    peft_layers = mod._decoder_layers(peft)
    assert len(peft_layers) == len(base_layers), (
        "PEFT-wrapped model must expose the same decoder layers as the base "
        "(the v6 smoke caught the .model.layers AttributeError on the adapter read)"
    )


def test_load_instruction_bank_failloud_on_empty(tmp_path: Path):
    """load_instruction_bank raises on an empty bank (a silent empty bank would
    yield a degenerate zero diff-mean direction)."""
    import json

    mod = _load_diffmean_module()
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="zero usable instructions"):
        mod.load_instruction_bank(str(empty), None)
    # a well-formed local JSONL bank loads + auto-detects the text column
    bank = tmp_path / "bank.jsonl"
    bank.write_text('{"prompt": "do harmful thing A"}\n{"goal": "do harmful thing B"}\n')
    rows = mod.load_instruction_bank(str(bank), None)
    assert rows == ["do harmful thing A", "do harmful thing B"]


# ── B2 (round-2): refusal K2 eval probes are HELD OUT from the extraction prompts ──


def test_split_train_eval_bank_disjoint(tmp_path: Path):
    """B2 regression: the K2 eval slice must be DISJOINT from the diff-mean extraction
    (train) slice. The prior code truncated the bank to n_train FIRST, then took the
    LAST k of that SAME slice as the K2 eval set -> the eval probes were a strict
    SUBSET of the extraction prompts (an in-sample sanity gate, not held out)."""
    mod = _load_diffmean_module()
    # unique rows so any overlap is unambiguous
    full = [f"row_{i:03d}" for i in range(40)]
    train, eval_rows = mod.split_train_eval_bank(
        full, n_train=20, n_eval_half=8, bank_name="harmful"
    )
    assert len(train) == 20 and len(eval_rows) == 8
    assert set(train).isdisjoint(eval_rows), "K2 eval slice overlaps the extraction prompts"
    # the eval slice is taken AFTER the train slice (genuinely held out, never extracted on)
    assert eval_rows == full[20:28]


def test_split_train_eval_bank_failloud_when_too_small(tmp_path: Path):
    """A bank too small to carve a DISJOINT held-out slice fails loud (never silently
    re-uses extraction rows for K2)."""
    mod = _load_diffmean_module()
    full = [f"row_{i}" for i in range(10)]
    with pytest.raises(ValueError, match=r"held-out K2 eval set cannot be carved|needs >="):
        mod.split_train_eval_bank(full, n_train=8, n_eval_half=4, bank_name="harmful")  # needs 12


def test_split_train_eval_bank_failloud_on_duplicate_rows(tmp_path: Path):
    """If the bank has duplicate rows that would put the SAME text in both train and
    eval, the disjointness assert fails loud rather than silently leaking a probe."""
    mod = _load_diffmean_module()
    # row_000 appears in both the would-be train slice (idx 0) and eval slice (idx 4)
    full = ["row_000", "row_001", "row_002", "row_003", "row_000", "row_005"]
    with pytest.raises(ValueError, match=r"overlap|duplicate"):
        mod.split_train_eval_bank(full, n_train=4, n_eval_half=2, bank_name="harmful")


def test_refusal_banks_train_eval_disjoint_on_real_local_banks(tmp_path: Path):
    """End-to-end B2: both refusal banks (harmful + harmless), loaded from local files
    of UNIQUE rows and split the way main() does, yield disjoint train/eval slices."""
    mod = _load_diffmean_module()
    harmful_bank = tmp_path / "harmful.jsonl"
    harmless_bank = tmp_path / "harmless.jsonl"
    harmful_bank.write_text("\n".join(f'{{"prompt": "harmful_{i:03d}"}}' for i in range(40)) + "\n")
    harmless_bank.write_text(
        "\n".join(f'{{"instruction": "harmless_{i:03d}"}}' for i in range(40)) + "\n"
    )
    n_train, n_eval_half = 16, 4
    for bank, name in ((harmful_bank, "harmful"), (harmless_bank, "harmless")):
        full = mod.load_instruction_bank(str(bank), None)
        train, eval_rows = mod.split_train_eval_bank(full, n_train, n_eval_half, name)
        assert set(train).isdisjoint(eval_rows), f"{name}: K2 eval overlaps extraction prompts"
        assert len(train) == n_train and len(eval_rows) == n_eval_half


# ── B4 (round-2): marker fallback FAILS LOUD on a primary crash, never demotes ──


def _marker_decision(marker_rc: int, effect_file: Path) -> subprocess.CompletedProcess:
    """Invoke the dispatcher's B4 marker-fallback decision hook in isolation
    (--test-marker-decision <rc> <effect-file>): no GPU, no upstream phases."""
    return subprocess.run(
        ["bash", str(DISPATCHER), "--test-marker-decision", str(marker_rc), str(effect_file)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_marker_fallback_fails_loud_on_primary_crash(tmp_path: Path):
    """B4 regression: a primary marker extraction that EXITS NONZERO before writing the
    effect file must FAIL LOUD (non-zero rc, no decision printed), NOT silently invoke
    the trained-shift fallback + M-Alts3 demotion. Pre-fix the dispatcher captured
    MARKER_RC but never read it, so any crash routed to the fallback."""
    proc = _marker_decision(1, tmp_path / "never_written.json")
    assert proc.returncode != 0, "a primary crash must fail loud, not return a decision"
    # crucially, the crash path NEVER emits the 'fallback' decision the dispatcher acts on
    assert "fallback" not in proc.stdout
    assert "affordance" not in proc.stdout
    assert "CRASH" in proc.stderr or "crash" in proc.stderr


def test_marker_fallback_fails_loud_on_missing_effect_file(tmp_path: Path):
    """B4: primary rc=0 but no effect file (a silent crash that left no K2 verdict) ->
    fail loud, no fallback."""
    proc = _marker_decision(0, tmp_path / "absent.json")
    assert proc.returncode != 0
    assert "fallback" not in proc.stdout and "affordance" not in proc.stdout


def test_marker_fallback_fails_loud_on_malformed_effect_file(tmp_path: Path):
    """B4: primary rc=0 + an effect file WITHOUT a boolean k2_pass -> fail loud (a
    malformed verdict must not be silently treated as a K2 failure)."""
    import json

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"foo": 1}))  # no k2_pass key
    proc = _marker_decision(0, bad)
    assert proc.returncode != 0
    assert "fallback" not in proc.stdout and "affordance" not in proc.stdout


def test_marker_fallback_invoked_only_on_real_k2_failure(tmp_path: Path):
    """B4 positive control: the trained-shift fallback fires when the primary wrote a
    valid effect JSON with k2_pass == false (here at rc=0, the artificial in-process
    case)."""
    import json

    eff = tmp_path / "eff.json"
    eff.write_text(json.dumps({"k2_pass": False}))
    proc = _marker_decision(0, eff)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "fallback"


def test_marker_fallback_decision_rc1_with_valid_k2_fail_falls_back(tmp_path: Path):
    """B4 round-3 regression (the PRODUCTION K2-HALT shape, the case round-2 missed):
    a primary that EXITS NONZERO but FIRST wrote a valid effect JSON with k2_pass=false
    must route to the #521 trained-shift fallback, NOT fail loud.

    This is the real shape: scripts/issue623_extract_sycophancy_vector.py raises
    SystemExit (rc != 0) AFTER writing steering_effect_by_layer_marker.json with
    k2_pass=false — that SystemExit IS the K2 HALT, not a crash. The round-2 code
    checked rc != 0 FIRST and returned 3 before reading the effect file, so a
    legitimate K2 fail (rc=1 + valid k2_pass=false) was mis-classified as a crash and
    the documented fallback never fired (launch attempt 5, 2026-06-18). The round-3 fix
    inverts the priority: route on the effect file's k2_pass boolean first; rc is only a
    fall-through crash signal when the effect file is unusable."""
    import json

    eff = tmp_path / "eff.json"
    eff.write_text(
        json.dumps({"k2_pass": False, "headline_layer": 7, "headline_layer_effect": 0.0})
    )
    proc = _marker_decision(1, eff)
    assert proc.returncode == 0, (
        f"rc=1 + valid k2_pass=false is the production K2 HALT and must fall back, "
        f"not fail loud (got rc={proc.returncode}, stderr={proc.stderr!r})"
    )
    assert proc.stdout.strip() == "fallback"


def test_marker_affordance_kept_on_k2_pass(tmp_path: Path):
    """B4 positive control: a real K2 PASS (rc=0, k2_pass=true) keeps marker PRIMARY
    (affordance), never the fallback."""
    import json

    eff = tmp_path / "eff.json"
    eff.write_text(json.dumps({"k2_pass": True}))
    proc = _marker_decision(0, eff)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "affordance"


# ── LIVE: real Claude artifact generation for marker (#623 recipe; API, no GPU) ──


def test_generate_artifacts_live_marker():
    """Real Claude call: generate_artifacts() for the MARKER trait must return a
    well-formed artifact dict (5 instruction pairs, >= n_questions questions,
    non-empty eval_prompt). Marker is the only production behavior still using the
    #623 generate_artifacts recipe in v6 (refusal/em use the diff-mean extractor).

    Skips loud when ANTHROPIC_API_KEY is absent so the keyless test sweep stays
    green; the implementer runs this locally with `.env` sourced before pushing.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — live artifact smoke requires it (source .env)")

    mod = _load_extract_module()
    from explore_persona_space.experiments.persona_decomp_623 import (
        DEFAULT_GENERATOR_MODEL,
        N_INSTRUCTION_PAIRS,
    )

    n_questions = 4  # tiny slice (production is 40); recipe still demands 5 pairs
    desc = _dispatcher_trait_description(MARKER_BEHAVIOR)
    data = mod.generate_artifacts(
        generator_model=DEFAULT_GENERATOR_MODEL,
        n_questions=n_questions,
        trait_name=MARKER_BEHAVIOR,
        trait_description=desc,
    )

    assert isinstance(data, dict)
    assert len(data["instruction"]) == N_INSTRUCTION_PAIRS, (
        f"marker: expected {N_INSTRUCTION_PAIRS} instruction pairs, got {len(data['instruction'])}"
    )
    for pair in data["instruction"]:
        assert "pos" in pair and "neg" in pair, "marker: instruction pair missing pos/neg"
        assert pair["pos"].strip() and pair["neg"].strip()
    assert len(data["questions"]) >= n_questions, (
        f"marker: expected >= {n_questions} questions, got {len(data['questions'])}"
    )
    assert data.get("eval_prompt"), "marker: empty eval_prompt"
    # The eval prompt must carry the {question}/{answer} placeholders the judge
    # substitution depends on (a malformed eval prompt silently breaks judging).
    assert "{question}" in data["eval_prompt"] and "{answer}" in data["eval_prompt"], (
        "marker: eval_prompt missing {question}/{answer} placeholders"
    )
