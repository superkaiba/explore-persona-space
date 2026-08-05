"""Pins for r14: scaffold staging verification + phase_b `--variants` scoping.

Covers (a) `scripts/issue2054_stage_scaffolds.py`'s verification gates —
count match, conv_id SET equality, duplicate/missing conv_id, undecodable
line, admission-record integrity — exercised as deliberate degenerate-input
probes (no network: the staging leg itself is smoke-covered against the real
Hub); and (b) `scripts/issue2054_phase_b.py --variants` — the plan §4 Cells
scoping flag (chat / bare_text are structurally assistant-only): scoped run,
unknown-variant fail-loud, and the default all-discovered behavior.

All fixtures are synthetic prose written for this test — no real-corpus
text. All writes go to pytest tmp_path (never canonical eval_results/ or
data/ paths).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_scaffold_common as sc  # noqa: E402
import issue2054_forms as forms  # noqa: E402
import issue2054_phase_b as phase_b  # noqa: E402
import issue2054_stage_scaffolds as stage  # noqa: E402

S = sc.SLOT_SENTINEL
QUESTION = "Where does the river go when the dam closes?"
ANSWER = "It pools behind the gate until the spillway opens."


# ---------------------------------------------------------------------------
# (a) stage_scaffolds verification gates (degenerate-input probes)
# ---------------------------------------------------------------------------
def _write_pool(path: Path, cids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid in cids:
            f.write(json.dumps({"conv_id": cid, "scaffold_id": f"sc_{cid}"}) + "\n")
    return path


def test_verify_staged_variant_ok(tmp_path):
    pool = _write_pool(tmp_path / "scaffolds_x.jsonl", ["a", "b", "c"])
    digest = stage.verify_staged_variant(pool, "x", ["a", "b", "c"])
    assert digest["rows"] == 3
    assert digest["bytes"] == pool.stat().st_size
    assert len(digest["sha256"]) == 64


def test_verify_count_mismatch_raises(tmp_path):
    pool = _write_pool(tmp_path / "scaffolds_x.jsonl", ["a", "b"])
    with pytest.raises(RuntimeError, match=r"row count 2 != .* admitted count 3"):
        stage.verify_staged_variant(pool, "x", ["a", "b", "c"])


def test_verify_set_mismatch_same_count_raises(tmp_path):
    pool = _write_pool(tmp_path / "scaffolds_x.jsonl", ["a", "b", "z"])
    with pytest.raises(RuntimeError, match="set mismatch"):
        stage.verify_staged_variant(pool, "x", ["a", "b", "c"])


def test_verify_duplicate_conv_id_raises(tmp_path):
    pool = _write_pool(tmp_path / "scaffolds_x.jsonl", ["a", "a", "b"])
    with pytest.raises(RuntimeError, match="duplicate conv_id"):
        stage.verify_staged_variant(pool, "x", ["a", "b", "c"])


def test_verify_missing_conv_id_raises(tmp_path):
    pool = tmp_path / "scaffolds_x.jsonl"
    pool.write_text(json.dumps({"scaffold_id": "sc_1"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no conv_id"):
        stage.verify_staged_variant(pool, "x", ["a"])


def test_verify_undecodable_line_raises(tmp_path):
    pool = tmp_path / "scaffolds_x.jsonl"
    pool.write_text('{"conv_id": "a"}\n{not json\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="undecodable"):
        stage.verify_staged_variant(pool, "x", ["a"])


def test_admitted_ids_for_gates():
    kept = {
        "variants": {
            "ok": {"admitted_conv_ids": ["a", "b"]},
            "empty": {"admitted_conv_ids": []},
            "dup": {"admitted_conv_ids": ["a", "a"]},
        }
    }
    assert stage.admitted_ids_for(kept, "ok") == ["a", "b"]
    with pytest.raises(RuntimeError, match="no admission record"):
        stage.admitted_ids_for(kept, "absent")
    with pytest.raises(RuntimeError, match="EMPTY"):
        stage.admitted_ids_for(kept, "empty")
    with pytest.raises(RuntimeError, match="duplicates"):
        stage.admitted_ids_for(kept, "dup")


# ---------------------------------------------------------------------------
# (b) phase_b --variants scoping
# ---------------------------------------------------------------------------
def _scaffold_row(cid: str, character: str) -> dict:
    text = f'{character} paused by the door. "{QUESTION}" the visitor asked. {S} Rain fell.'
    return {
        "scaffold_id": f"sc_{cid}",
        "conv_id": cid,
        "character": character,
        "question": QUESTION,
        "scaffold_text": text,
    }


def _build_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Two-variant scaffold tree + an answers pool covering every conv_id."""
    root = tmp_path / "scaffolds"
    rows = {
        "char_vex": [_scaffold_row("v1", "Vex"), _scaffold_row("v2", "Vex")],
        "conversation_paired_stories_assistant": [_scaffold_row("a1", "Assistant")],
    }
    for variant, vrows in rows.items():
        vdir = root / variant
        vdir.mkdir(parents=True)
        with (vdir / f"scaffolds_{variant}.jsonl").open("w", encoding="utf-8") as f:
            for r in vrows:
                f.write(json.dumps(r) + "\n")
    answers = tmp_path / "answers_pool.jsonl"
    with answers.open("w", encoding="utf-8") as f:
        for cid in ("v1", "v2", "a1"):
            f.write(json.dumps({"conv_id": cid, "answer": ANSWER}) + "\n")
    return root, answers


def _ns(root: Path, answers: Path, out: Path, form: str, variants: str | None):
    return argparse.Namespace(
        scaffolds_dir=str(root),
        answers_source=str(answers),
        form=form,
        output_dir=str(out),
        seed=137,
        skip_upload=True,
        variants=variants,
    )


def _out_file(out: Path, variant: str, form: str) -> Path:
    return out / variant / forms.phase_output_name("inserted", variant, form)


def test_variants_scopes_chat_to_assistant(tmp_path):
    root, answers = _build_tree(tmp_path)
    out = tmp_path / "out_chat"
    ns = _ns(root, answers, out, "chat", "conversation_paired_stories_assistant")
    with pytest.raises(SystemExit) as ei:
        phase_b.run_phase(ns)
    assert ei.value.code == 0
    asst = _out_file(out, "conversation_paired_stories_assistant", "chat")
    assert asst.is_file()
    spliced = [json.loads(x) for x in asst.read_text(encoding="utf-8").split("\n") if x.strip()]
    assert len(spliced) == 1 and spliced[0]["form"] == "chat"
    # The character variant was NOT rendered (plan-excluded cell).
    assert not _out_file(out, "char_vex", "chat").exists()
    assert not (out / "char_vex").exists()


def test_variants_unknown_fails_loud(tmp_path):
    root, answers = _build_tree(tmp_path)
    ns = _ns(root, answers, tmp_path / "out_bad", "chat", "no_such_variant")
    assert phase_b.run_phase(ns) == 1


def test_variants_default_runs_all_discovered(tmp_path):
    root, answers = _build_tree(tmp_path)
    out = tmp_path / "out_story"
    ns = _ns(root, answers, out, "attrib_quoted", None)
    with pytest.raises(SystemExit) as ei:
        phase_b.run_phase(ns)
    assert ei.value.code == 0
    for variant, n in (("char_vex", 2), ("conversation_paired_stories_assistant", 1)):
        f = _out_file(out, variant, "attrib_quoted")
        assert f.is_file()
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").split("\n") if x.strip()]
        assert len(rows) == n
        assert all(r["answer_source"] == "answers_pool" for r in rows)
