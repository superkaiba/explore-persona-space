"""Pins for the #2054 phase_d cell (c) driver (`scripts/issue2054_phase_d.py`).

Round-2 findings C4 + C5:

C4 — cell (c) is "answer authored STORY, presented CHAT" (plan §4 Block 3 /
Phase D): the render must go through the CHAT template (plan framing 1), never
share the story presentation with cell (d), and the dispatch wire must PASS
`--form chat` rather than merely documenting it.

C5 — the phase_a→phase_d conv_id join crosses two key spaces: production
scaffolds carry `conv_id = scaffold_id = f"stripped_{story_id}"` (the parent
stripper via phase_a's setdefault) while parent paired_op story rows key on
the bare `story_id`. The join (and the shared fold-map membership check)
canonizes both sides via `_canon_conv_id`; the acceptance test runs the REAL
producer (`issue1345_strip_scaffolds.strip_file`) → phase_d `main()` WITHOUT
`--no-fold-filter` and requires exit 0 with n_out > 0.

Only the HF network boundary is faked (signature-conformant fakes for
`_list_parent_story_files` / `_download_parent_story`); every other body —
strip_file, _index_parent_answers, _extract_op_answer, _process_variant,
_splice_one, run_phase — executes for real. All fixtures are synthetic prose
written for this test — no real-corpus text.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_scaffold_common as sc  # noqa: E402
import issue1345_strip_scaffolds as strip_cli  # noqa: E402
import issue2054_forms as forms  # noqa: E402
import issue2054_phase_d as phase_d  # noqa: E402

# Two single-turn stories in the parent's parseable shape ("<Name> replied:"
# marker ends with ":" -> confidence.marker_exact; answers 20..2000 chars ->
# confidence.answer_len_ok — the `_extract_op_answer` keep conditions).
STORY_1 = (
    'Mira leaned over the rail. "Where does the river go when the dam closes?" '
    'she asked. Helios replied: "It pools in the old quarry until the gates '
    'reopen." The wind picked up.'
)
STORY_2 = (
    'The technician tapped the gauge. "Is the reactor loop holding pressure '
    'tonight?" she asked. Helios replied: "The core temperature is stable and '
    'the loop holds through the night." Snow kept falling outside.'
)
ANSWER_1 = "It pools in the old quarry until the gates reopen."


def _stripped_scaffold_rows(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Parent stories JSONL + phase_a-shaped scaffold rows via the REAL producers.

    Returns (parent_jsonl_path, scaffold_rows). The conv_id convention is the
    exact phase_a one (`issue2054_phase_a.py` `_recover_scaffolds`:
    `row.setdefault("conv_id", row.get("scaffold_id", ...))`).
    """
    parent = tmp_path / "x_stories_paired_op_1.jsonl"
    rows = [
        {"story_id": "s0001", "conv_id": "s0001", "story": STORY_1},
        {"story_id": "s0002", "conv_id": "s0002", "story": STORY_2},
    ]
    parent.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    scaffold_rows, counts = strip_cli.strip_file(parent, "Helios")
    assert counts["kept"] == 2, counts
    for i, row in enumerate(scaffold_rows):
        row.setdefault("conv_id", row.get("scaffold_id", f"char_helios_{i}"))
        row.setdefault("variant", "char_helios")
    # Pin the PRODUCTION key shape the C5 canonization exists for: the
    # stripper prefixes story_id, so a scaffold conv_id is `stripped_s...`
    # (the committed shared_fold_map.json keys have exactly this shape).
    assert scaffold_rows[0]["conv_id"] == "stripped_s0001", scaffold_rows[0]["conv_id"]
    return parent, scaffold_rows


# ---------------------------------------------------------------------------
# C5 — canonical join key
# ---------------------------------------------------------------------------
def test_canon_conv_id_strips_stripper_prefix_and_is_idempotent():
    assert phase_d._canon_conv_id("stripped_s2445") == "s2445"
    assert phase_d._canon_conv_id("s2445") == "s2445"
    assert phase_d._canon_conv_id(phase_d._canon_conv_id("stripped_s2445")) == "s2445"


def test_index_parent_answers_keys_on_canonical_conv_id(tmp_path, monkeypatch):
    parent, _ = _stripped_scaffold_rows(tmp_path)

    def fake_download(path_in_repo: str) -> Path | None:
        return parent

    monkeypatch.setattr(phase_d, "_download_parent_story", fake_download)
    answers = phase_d._index_parent_answers(["fake/path.jsonl"], "Helios")
    assert answers["s0001"] == ANSWER_1
    assert set(answers) == {"s0001", "s0002"}


# ---------------------------------------------------------------------------
# C4 — cell (c) presentation: CHAT template, not the story presentation
# ---------------------------------------------------------------------------
def test_cell_c_chat_render_not_story_presentation(tmp_path):
    _, scaffold_rows = _stripped_scaffold_rows(tmp_path)
    row = scaffold_rows[0]
    answer = "The gates reopen at first light tomorrow."

    out = phase_d._splice_one(row, answer, "Helios", "char_helios_op", "chat")
    assert out is not None
    text = out["final_text"]
    # The plan's chat markers (framing 1) appear in final_text ...
    assert text.startswith(forms.CHAT_USER_HEADER)
    assert forms.CHAT_ASSISTANT_HEADER in text
    assert text.endswith(forms.CHAT_SUFFIX)
    assert out["form"] == "chat"
    assert text[out["answer_start"] : out["answer_end"]] == answer
    assert out["prefix_end_char"] == len(forms.CHAT_USER_HEADER)
    # ... and the STORY presentation does not (no attribution clause, no
    # narrative prose, no slot sentinel): authorship and presentation stay
    # separable — the point of the v4 2x2.
    assert 'replied: "' not in text
    assert "leaned over the rail" not in text
    assert sc.SLOT_SENTINEL not in text

    # Contrast: the SAME row under the story form (cell (d)'s presentation)
    # DOES carry the attribution shape — the presentation term C4 separates.
    out_d = phase_d._splice_one(row, answer, "Helios", "char_helios", "attrib_quoted")
    assert out_d is not None
    assert 'Helios replied: "' in out_d["final_text"]


def test_dispatch_plan_pins_form_chat_for_phase_d():
    res = subprocess.run(
        ["bash", str(_REPO_ROOT / "scripts" / "issue2054_dispatch.sh"), "--plan", "phase_d"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert res.returncode == 0, res.stderr
    assert "--form chat" in res.stdout, res.stdout


# ---------------------------------------------------------------------------
# C5 acceptance — phase_a -> phase_d e2e WITHOUT --no-fold-filter
# ---------------------------------------------------------------------------
def test_phase_a_to_phase_d_join_e2e_without_fold_bypass(tmp_path, monkeypatch):
    parent, scaffold_rows = _stripped_scaffold_rows(tmp_path)

    scaff_dir = tmp_path / "scaffolds" / "char_helios"
    scaff_dir.mkdir(parents=True)
    (scaff_dir / "scaffolds_char_helios.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scaffold_rows),
        encoding="utf-8",
    )

    # Shared fold map in the COMMITTED production key shape (scaffold-space,
    # stripper-prefixed). s0002 deliberately absent -> exercises the filter.
    fold_map = tmp_path / "shared_fold_map.json"
    fold_map.write_text(json.dumps({"fold_of": {"stripped_s0001": 0}}), encoding="utf-8")

    # Fake ONLY the HF network boundary (signature-conformant).
    def fake_list(api, variant: str) -> list[str]:
        return [f"issue1345_framing/{variant}/raw_completions/stories/x_stories_paired_op_1.jsonl"]

    def fake_download(path_in_repo: str) -> Path | None:
        return parent

    monkeypatch.setattr(phase_d, "_list_parent_story_files", fake_list)
    monkeypatch.setattr(phase_d, "_download_parent_story", fake_download)

    out_dir = tmp_path / "cell_c"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue2054_phase_d.py",
            "--scaffolds-dir",
            str(tmp_path / "scaffolds"),
            "--output-dir",
            str(out_dir),
            "--form",
            "chat",
            "--variants",
            "char_helios_op",
            "--fold-map",
            str(fold_map),
            "--skip-upload",
        ],
    )
    # NO --no-fold-filter: the join must resolve against the real fold map.
    # Pre-C5-fix this never raises (run_phase returns 1 on zero spliced rows).
    with pytest.raises(SystemExit) as exc:
        phase_d.main()
    assert exc.value.code == 0

    # Form-aware output name (C6): the cell (c) file embeds the --form axis.
    out_path = (
        out_dir / "char_helios_op" / forms.phase_output_name("cell_c", "char_helios_op", "chat")
    )
    assert out_path.name == "cell_c_char_helios_op__chat.jsonl"
    out_rows = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    assert len(out_rows) == 1  # s0001 in fold; s0002 filtered out-of-fold
    row = out_rows[0]
    assert row["form"] == "chat"
    assert row["final_text"].startswith(forms.CHAT_USER_HEADER)
    assert forms.CHAT_ASSISTANT_HEADER in row["final_text"]
    assert row["answer"] == ANSWER_1
    assert row["final_text"][row["answer_start"] : row["answer_end"]] == ANSWER_1
    # Raw scaffold key kept for cross-cell conv matching; canonical parent key
    # recorded for provenance.
    assert row["conv_id"] == "stripped_s0001"
    assert row["parent_conv_id"] == "s0001"

    # Form-keyed digest name (C6): one digest per (condition, form) run.
    digest = json.loads((out_dir / "phase_d_digest__chat.json").read_text(encoding="utf-8"))
    assert digest["n_total_out"] == 1
    assert digest["form"] == "chat"
    assert digest["fold_map"] == str(fold_map.resolve())
    per = digest["per_variant"][0]
    assert per["n_out_of_fold"] == 1
    assert per["n_no_answer"] == 0  # the join resolves for every in-fold row


def test_missing_fold_map_fails_loud_without_bypass(tmp_path, monkeypatch, capsys):
    scaff_root = tmp_path / "scaffolds"
    scaff_root.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue2054_phase_d.py",
            "--scaffolds-dir",
            str(scaff_root),
            "--output-dir",
            str(tmp_path / "cell_c"),
            "--form",
            "chat",
            "--fold-map",
            str(tmp_path / "nonexistent_fold_map.json"),
            "--skip-upload",
        ],
    )
    rc = phase_d.main()
    assert rc == 1
    assert "shared fold map not found" in capsys.readouterr().err
