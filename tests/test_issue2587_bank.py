"""CPU pins for the issue #2587 merged bank (plan v3 §4.4).

No network, no HF fetch, no GPU: the q35 tokenizer is FAKED at the
chat-template boundary (the ``test_issue2333_run_units.py``
``_FakeTemplateTok`` pattern) with a deterministic word-level id map, so the
full 1,080-context / 2,874-pair build + every gate runs in-process. Pins:

- the plan §4.4 arithmetic (984 + 96 contexts; 2,778 + 96 pairs; exact
  per-class counts incl. the pilot classes 36/36/24);
- gate fail-loud vs recorded vs reported dispositions EXACTLY as the plan
  splits them (iii/iv record + report and never raise; i/ii/v/vi/vii raise);
- the VERBATIM transcription of the langow pilot-axis definitions against
  the pinned source commit (AST literal compare + sha256);
- the pinned-blob sha256 constants against ``git show`` ground truth.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
import subprocess
import zlib
from pathlib import Path

import pytest

from explore_persona_space.experiments.issue2587 import bank2587 as B

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_show_bytes(ref: str, rel: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{ref}:{rel}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout


class FakeQ35Tok:
    """q35-shaped chat-template fake: Qwen-style role blocks + a generation
    prompt ending in the thinking-off block; deterministic word-level ids
    (crc32 — PYTHONHASHSEED-independent) so ``changed_token_count`` is real."""

    name_or_path = "fake-q35"

    def __init__(self, think_block: str = "<think>\n\n</think>\n\n", extra_suffix: str = ""):
        self.think_block = think_block
        self.extra_suffix = extra_suffix

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        assert kw.get("enable_thinking") is False, "render must pass enable_thinking=False"
        assert add_generation_prompt is True and tokenize is False
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return body + "<|im_start|>assistant\n" + self.think_block + self.extra_suffix

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        toks = re.findall(r"\S+|\n", text)
        return {"input_ids": [zlib.crc32(t.encode("utf-8")) for t in toks]}


class ConstantRenderTok(FakeQ35Tok):
    """Renders EVERY context identically — trips gate (v) (changed_tokens >= 1)
    while passing the render gate (vii)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        assert kw.get("enable_thinking") is False
        return "<|im_start|>user\nconstant<|im_end|>\n<|im_start|>assistant\n" + self.think_block


class CharTok(FakeQ35Tok):
    """Character-level ids: '' {NAME}'' encodes to MANY tokens — exercises the
    gate (iv) record-not-assert disposition."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


@pytest.fixture(scope="module")
def bank():
    return B.build_bank(FakeQ35Tok())


@pytest.fixture(scope="module")
def values():
    return B._bk().load_values()


# ── plan §4.4 arithmetic ───────────────────────────────────────────────


def test_expected_totals_constants():
    assert B.N_CONTEXTS == 1080 == 984 + 96
    assert B.N_PAIRS == 2874 == 2778 + 96
    assert sum(B.EXPECTED_PAIR_COUNTS.values()) == 2874
    assert sum(B.EXPECTED_PILOT_PAIR_COUNTS.values()) == 96


def test_merged_counts(bank):
    assert bank["n_contexts"] == len(bank["contexts"]) == 1080
    assert bank["n_pairs"] == len(bank["pairs"]) == 2874
    counts: dict[str, int] = {}
    for p in bank["pairs"]:
        counts[p["pair_class"]] = counts.get(p["pair_class"], 0) + 1
    assert counts == {
        "install": 504,
        "swap": 900,
        "famswap": 864,
        "instruction_paraphrase": 468,
        "query_content": 66,
        "query_form": 36,
        "query_paraphrase": 12,
        "query_content_oneword": 24,
    }
    assert len({p["pair_id"] for p in bank["pairs"]}) == 2874
    assert {len(v) for v in bank["per_cell_pilot"].values()} == {48}
    by_cell: dict[str, int] = {}
    for ctx in bank["contexts"].values():
        by_cell[ctx["cell"]] = by_cell.get(ctx["cell"], 0) + 1
    assert by_cell["answer_language"] == 48
    assert by_cell["query_content_oneword"] == 48
    assert sum(n for c, n in by_cell.items() if c not in B.PILOT_CELLS) == 984


def test_pilot_pair_conventions(bank):
    pilot = [p for p in bank["pairs"] if p.get("axis") in B.PILOT_CELLS]
    assert len(pilot) == 96
    by_class: dict[str, int] = {}
    for p in pilot:
        by_class[p["pair_class"]] = by_class.get(p["pair_class"], 0) + 1
        assert p["cell"] == p["axis"]  # parent-schema grouping key present
    assert by_class == {"install": 36, "swap": 36, "query_content_oneword": 24}
    contexts = bank["contexts"]
    for p in pilot:
        a, b = contexts[p["a"]], contexts[p["b"]]
        if p["pair_class"] == "install":
            # langow orientation: a = language-value context, b = bare.
            assert a["system"] in B.LANG_VALUES.values() and b["system"] == ""
            assert p["value_b"] == "bare"
        elif p["pair_class"] == "swap":
            assert a["system"] != b["system"]
            assert {a["value_id"], b["value_id"]} <= set(B.LANG_VALUES)
        else:
            assert a["system"] == b["system"] == ""
            assert p["value_a"].endswith("a") and p["value_b"].endswith("b")
        assert a["carrier"] == b["carrier"] == p["carrier"]
    # swap pairs follow LANG_VALUES insertion order i < j.
    langs = tuple(B.LANG_VALUES)
    for p in pilot:
        if p["pair_class"] == "swap":
            assert langs.index(p["value_a"]) < langs.index(p["value_b"]), p["pair_id"]


def test_context_id_conventions(bank):
    assert "answer_language::bare::c01" in bank["contexts"]
    assert "answer_language::english::c12" in bank["contexts"]
    assert "query_content_oneword::p01a::c01" in bank["contexts"]
    assert "query_content_oneword::p24b::c12" in bank["contexts"]
    ids = {p["pair_id"] for p in bank["pairs"]}
    assert "install::answer_language::english-bare::c01" in ids
    assert "swap::answer_language::english-chinese::c01" in ids
    assert "query_content_oneword::query_content_oneword::p01a-p01b::c01" in ids


# ── gate dispositions: fail-loud gates raise ───────────────────────────


def test_gate_i_trips_on_missing_context(bank):
    contexts = dict(bank["contexts"])
    contexts.pop("answer_language::bare::c01")
    with pytest.raises(RuntimeError, match=r"gate\(i\)"):
        B.gate_merged_complete(contexts, bank["pairs"])


def test_gate_i_trips_on_missing_pair(bank):
    pairs = [p for p in bank["pairs"] if p["pair_id"] != "query_paraphrase::query::E-qpara::c01"]
    with pytest.raises(RuntimeError, match=r"gate\(i\)"):
        B.gate_merged_complete(bank["contexts"], pairs)


def test_gate_ii_trips_on_mutated_user(bank):
    contexts = copy.deepcopy(bank["contexts"])
    pilot_pairs = [p for p in bank["pairs"] if p.get("axis") == "answer_language"]
    contexts[pilot_pairs[0]["a"]]["user"] = "mutated question?"
    with pytest.raises(RuntimeError, match=r"gate\(ii\).*user strings differ"):
        B.gate_pilot_slot_identity(contexts, pilot_pairs)


def test_gate_vi_trips_on_assistant_in_system(bank):
    contexts = copy.deepcopy(bank["contexts"])
    contexts["answer_language::english::c01"]["system"] = "You are an assistant."
    with pytest.raises(RuntimeError, match=r"gate\(vi\)"):
        B.gate_no_assistant_in_system_strings(contexts)


def test_gate_vi_form_triplets_trip_via_parent(values):
    broken = copy.deepcopy(values)
    broken["carriers"]["c01"]["imperative"] = "I'm unsure — tell me whether to adopt a dog or cat."
    with pytest.raises(RuntimeError, match=r"gate\(vii\).*affect term"):
        # the pinned parent's gate numbering calls form-triplets (vii);
        # plan §4.4 folds it into gate (vi) — either way it must raise.
        B.build_bank_strings(broken)


def test_gate_v_trips_on_identical_renders(bank):
    strings_bank = B.build_bank_strings()
    with pytest.raises(AssertionError, match="render identically"):
        B.run_token_gates(ConstantRenderTok(), strings_bank)


def test_gate_vii_trips_on_open_think():
    strings_bank = B.build_bank_strings()
    with pytest.raises(AssertionError, match="OPEN thinking block"):
        B.run_token_gates(FakeQ35Tok(think_block="<think>\n"), strings_bank)


def test_gate_vii_trips_on_nonempty_think():
    strings_bank = B.build_bank_strings()
    with pytest.raises(AssertionError, match="non-empty thinking block"):
        B.run_token_gates(
            FakeQ35Tok(think_block="<think>\nreasoning...\n</think>\n\n"), strings_bank
        )


def test_gate_vii_trips_on_double_assistant_header():
    strings_bank = B.build_bank_strings()
    with pytest.raises(RuntimeError, match=r"gate\(vii\).*headers != 1"):
        B.run_token_gates(FakeQ35Tok(extra_suffix="<|im_start|>assistant\n"), strings_bank)


# ── gate dispositions: recorded/reported gates NEVER raise ─────────────


def test_gate_iii_records_and_reports_without_assert(bank):
    tg = bank["token_gates"]
    assert tg["verdict"] == "PASS"
    assert tg["gates_run"] == ["iii", "iv", "v", "vii"]
    counts = tg["value_token_counts"]
    assert set(counts) == set(B._bk().INSTRUCTION_AXES) | {"answer_language"}
    assert len(counts["answer_language"]) == 3
    # within-axis equality is a REPORT (bool per axis), never an assert: the
    # fake word-level tokenizer yields unequal counts on several axes and the
    # build still PASSed.
    within = tg["within_axis_equal"]
    assert set(within) == set(counts)
    assert all(isinstance(v, bool) for v in within.values())
    assert not all(within.values()), "fixture should exhibit at least one unequal axis"
    assert tg["q25_expected_value_tokens"] == dict(B._bk().EXPECTED_VALUE_TOKENS)
    assert set(tg["paraphrase_token_counts"]) == set(B._bk().INSTRUCTION_AXES)


def test_gate_iv_records_multitoken_names_without_assert(values):
    rec = B.record_name_token_counts(CharTok(), values)
    assert set(rec) == set(B._bk().NAME_TOKEN_IDS)
    for name, row in rec.items():
        assert row["n_tokens"] == len(" " + name) > 1  # char-level: many tokens, NO raise
        assert row["single_token"] is False
        assert row["q25_pinned_id"] == B._bk().NAME_TOKEN_IDS[name]


def test_gate_v_changed_tokens_attached(bank):
    assert all(p["changed_tokens"] >= 1 for p in bank["pairs"])
    assert bank["token_gates"]["changed_tokens_min"] >= 1
    # oneword pairs differ in exactly one content word: small edit dose.
    oneword = [p for p in bank["pairs"] if p["pair_class"] == "query_content_oneword"]
    assert all(1 <= p["changed_tokens"] <= 4 for p in oneword)


def test_strings_only_build_runs_no_token_gates():
    strings_bank = B.build_bank_strings()
    assert strings_bank["token_gates"] is None
    assert strings_bank["string_gates"] == {"verdict": "PASS", "gates_run": ["i", "ii", "vi"]}
    assert "changed_tokens" not in strings_bank["pairs"][0]


# ── transcription + pin fidelity ───────────────────────────────────────


def test_langow_source_sha256_pin():
    data = _git_show_bytes(B.LANGOW_COMMIT, "scripts/issue2564_langow_pilot_run.py")
    assert hashlib.sha256(data).hexdigest() == B.LANGOW_SHA256


def test_pilot_definitions_transcribed_verbatim():
    """AST literal compare of LANG_VALUES + ONEWORD_PAIRS against the pinned
    langow source — the transcription can never drift silently."""
    src = _git_show_bytes(B.LANGOW_COMMIT, "scripts/issue2564_langow_pilot_run.py").decode()
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in ("LANG_VALUES", "ONEWORD_PAIRS")
            and node.value is not None
        ):
            found[node.target.id] = ast.literal_eval(node.value)
    assert set(found) == {"LANG_VALUES", "ONEWORD_PAIRS"}
    assert found["LANG_VALUES"] == B.LANG_VALUES
    assert tuple(found["LANG_VALUES"]) == tuple(B.LANG_VALUES)  # insertion order too
    assert found["ONEWORD_PAIRS"] == B.ONEWORD_PAIRS


def test_pinned_blob_sha256_ground_truth():
    for rel, want in B.PINNED_SHA256.items():
        data = _git_show_bytes(B.PIN, rel)
        assert hashlib.sha256(data).hexdigest() == want, rel


def test_committed_values_copy_matches_pin():
    committed = B.VALUES_PINNED_COPY.read_bytes()
    blob = _git_show_bytes(
        B.PIN, "src/explore_persona_space/experiments/issue2564/bank2564_values.json"
    )
    assert committed == blob


def test_pinned_bytes_sha_drift_fails(tmp_path, monkeypatch):
    bad = tmp_path / "bank2564_values_pinned.json"
    bad.write_bytes(b'{"tampered": true}')
    monkeypatch.setattr(B, "VALUES_PINNED_COPY", bad)
    with pytest.raises(RuntimeError, match="sha256"):
        B._pinned_bytes("src/explore_persona_space/experiments/issue2564/bank2564_values.json")


# ── manifest + misc ────────────────────────────────────────────────────


def test_write_bank_manifest(tmp_path, bank):
    out = tmp_path / "bank_manifest.json"
    B.write_bank_manifest(bank, out)
    import json

    manifest = json.loads(out.read_text())
    assert manifest["n_contexts"] == 1080 and manifest["n_pairs"] == 2874
    assert manifest["parent_pin"] == B.PIN
    assert manifest["langow_commit"] == B.LANGOW_COMMIT
    assert "metadata" in manifest and "timestamp_utc" in manifest["metadata"]


def test_pinned_callee_signatures():
    """Kwarg-signature smokes for the pinned parent callees bank2587 uses."""
    import inspect

    bk = B._bk()
    inspect.signature(bk.attach_changed_tokens).bind(pairs=[], ids_by_context={})
    inspect.signature(bk.gate_grid_complete).bind(values={}, contexts={}, pairs=[])
    inspect.signature(bk.gate_pair_slot_identity).bind(values={}, contexts={}, pairs=[])
    inspect.signature(bk.gate_form_triplets).bind(values={})
    inspect.signature(bk.context_id).bind("cell", "vid", "c01")
    inspect.signature(bk.pair_id).bind("cls", "cell", "va", "vb", "c01")
    inspect.signature(bk.write_bank_manifest).bind(bank={}, out_path=Path("x"))
    assert tuple(f"c{i:02d}" for i in range(1, 13)) == bk.CARRIER_IDS


def test_model_constants():
    assert B.MODEL_ID == "Qwen/Qwen3.5-9B"
    assert B.HIDDEN == 4096
    assert B.N_LAYERS == 32
