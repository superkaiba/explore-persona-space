"""#1738 r5 crash fix: kresample admission gate (over-length + no-primary-draw).

Pins the permanent invariant added after the attempt-3 kresample crash (an
UNGATED 14,217-token subsample row reached vLLM's ``llm_engine.add_request``
against ``max_model_len`` 8,192 and killed the engine): K-resample rows pass a
capture-parity admission gate — over-budget renders and rows without a primary
seed-42 draw are SKIP+RECORDED (ci + n_tokens + reasons, never text), never
engine-fatal. Pre-fix these symbols do not exist and no gate ran.

Production-body tests (code-style § one production-body test per seam-stubbed
function): the REAL helper bodies execute; ONLY the external boundaries are
faked signature-conformantly — the tokenizer boundary via a ``def`` mirroring
the ``tok_len_fn(messages) -> int`` contract, the Hub boundary via
``unittest.mock.create_autospec(hub._upload_folder_filtered)``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue1738_multiturn_generate_capture as GG  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402


def _row(i: int) -> dict:
    return {"i": i, "messages": [{"role": "user", "content": f"synthetic row {i}"}]}


def _len_fn(over: set[int]):
    """Signature-conformant fake at the tokenize boundary: tok_len_fn(messages)->int."""

    def _n_tokens(messages: list[dict]) -> int:
        i = int(messages[0]["content"].rsplit(" ", 1)[-1])
        return GG.PROMPT_TOKEN_BUDGET + 1 if i in over else 5

    return _n_tokens


def test_engine_cap_invariant():
    """Budget-admitted prompt + max generation always fits the engine."""
    assert GG.PROMPT_TOKEN_BUDGET + GG.GEN_MAX_TOKENS <= GG.MAX_MODEL_LEN


def test_admission_gate_overlength_and_primary():
    rows = [_row(i) for i in range(4)]
    kept, skipped = GG._kresample_admission_gate(rows, _len_fn({2}), primary_cis={0, 1, 2})
    assert [r["i"] for r in kept] == [0, 1]
    by_ci = {d["ci"]: d for d in skipped}
    assert by_ci[2]["reasons"] == ["overlength"]
    assert by_ci[2]["n_tokens"] == GG.PROMPT_TOKEN_BUDGET + 1
    assert by_ci[3]["reasons"] == ["no_primary_draw"]
    # a row failing BOTH gates records both reasons
    kept2, skipped2 = GG._kresample_admission_gate(rows, _len_fn({3}), primary_cis={0, 1, 2})
    assert {d["ci"]: d["reasons"] for d in skipped2}[3] == ["overlength", "no_primary_draw"]
    assert [r["i"] for r in kept2] == [0, 1, 2]


def test_admission_gate_primary_none_disables_membership_leg():
    rows = [_row(i) for i in range(3)]
    kept, skipped = GG._kresample_admission_gate(rows, _len_fn({1}), primary_cis=None)
    assert [r["i"] for r in kept] == [0, 2]
    assert [d["ci"] for d in skipped] == [1] and skipped[0]["reasons"] == ["overlength"]


def test_write_kresample_skipped_record_and_upload(monkeypatch, tmp_path):
    """Real writer body: record shape on disk + exact-set verified Hub upload."""
    fake = create_autospec(hub._upload_folder_filtered, return_value="repo/prefix")
    monkeypatch.setattr(GG.hub, "_upload_folder_filtered", fake)
    args = SimpleNamespace(
        shard_index=0,
        num_shards=1,
        no_upload=False,
        hf_prefix="issueTEST_mt",
        kresample_primary_ci="hf",
    )
    skipped = [{"ci": 98764, "n_tokens": 14217, "reasons": ["overlength"]}]
    GG._write_kresample_skipped(tmp_path, args, skipped)
    rec = json.loads((tmp_path / "kresample_shard00_skipped.json").read_text())
    assert rec["n_skipped"] == 1 and rec["skipped"] == skipped
    assert rec["prompt_token_budget"] == GG.PROMPT_TOKEN_BUDGET and rec["primary_gate"] is True
    kw = fake.call_args.kwargs
    assert kw["allow_patterns"] == ["kresample_shard00_skipped.json"]
    assert kw["expected_repo_paths"] == ["issueTEST_mt/kresample/kresample_shard00_skipped.json"]


def test_write_kresample_skipped_no_upload_and_fail_loud(monkeypatch, tmp_path):
    fake = create_autospec(hub._upload_folder_filtered, return_value="repo/prefix")
    monkeypatch.setattr(GG.hub, "_upload_folder_filtered", fake)
    args = SimpleNamespace(
        shard_index=1,
        num_shards=2,
        no_upload=True,
        hf_prefix="issueTEST_mt",
        kresample_primary_ci="none",
    )
    GG._write_kresample_skipped(tmp_path, args, [])
    assert fake.call_count == 0
    rec = json.loads((tmp_path / "kresample_shard01_skipped.json").read_text())
    assert rec["n_skipped"] == 0 and rec["primary_gate"] is False
    # empty upload URL fails loud (never a silent record loss)
    fake.return_value = ""
    args2 = SimpleNamespace(
        shard_index=1,
        num_shards=2,
        no_upload=False,
        hf_prefix="issueTEST_mt",
        kresample_primary_ci="hf",
    )
    with pytest.raises(RuntimeError, match="skipped-sidecar upload"):
        GG._write_kresample_skipped(tmp_path, args2, [])


def test_primary_cis_local_npz_and_none(tmp_path):
    np = pytest.importorskip("numpy")
    p = tmp_path / "primary.npz"
    np.savez(p, ci=np.asarray([3, 5, 8], dtype=np.int64))
    args = SimpleNamespace(kresample_primary_ci=str(p))
    assert GG._kresample_primary_cis(args) == {3, 5, 8}
    args_json = SimpleNamespace(kresample_primary_ci=str(tmp_path / "primary.json"))
    (tmp_path / "primary.json").write_text(json.dumps({"ci": [1, 2]}))
    assert GG._kresample_primary_cis(args_json) == {1, 2}
    assert GG._kresample_primary_cis(SimpleNamespace(kresample_primary_ci="none")) is None
