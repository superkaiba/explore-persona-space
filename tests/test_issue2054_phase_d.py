"""Pins for the #2054 phase_d cell (c) driver (`scripts/issue2054_phase_d.py`).

Round-2 findings C4 + C5:

C4 — cell (c) is "answer authored STORY, presented CHAT" (plan §4 Block 3 /
Phase D): the render must go through the CHAT template (plan framing 1), never
share the story presentation with cell (d), and the dispatch wire must PASS
`--form chat` rather than merely documenting it.

C5 — the phase_a→phase_d conv_id join canonizes both sides via
`_canon_conv_id` (stripper-path scaffold ids read `stripped_<story_id>`;
the committed shared_fold_map.json keys carry the same shape). The
acceptance test runs the REAL producer (`issue1345_strip_scaffolds
.strip_file`) → phase_d `main()` WITHOUT `--no-fold-filter` and requires
exit 0 with n_out > 0.

2026-08-06 answer-source rewire (USER DIRECTIVE "make sure we only use the
new data"; concern `cell-c-source-tonight-on-policy-not-parent-pool`): the
cell-(c) answers come from THIS task's Phase-C on-policy pool
(`issue2054_lattice/on_policy/{model}/{variant}/on_policy_{variant}__{form}
.jsonl`), NEVER the parent #1345 `stories_paired_op` pool. The pool rows
carry the answer directly (`issue2054_phase_c._splice_generated` schema).

Only the HF network boundary is faked (a signature-conformant fake for
`_download_pool_file`); every other body — strip_file, _index_pool_answers,
_process_variant, _splice_one, run_phase, the resume/regime machinery —
executes for real. All fixtures are synthetic prose written for this test —
no real-corpus text.
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

# Two single-turn stories in the stripper's parseable shape ("<Name> replied:"
# attribution marker) — the REAL `strip_file` producer turns these into
# phase_a-shaped scaffold rows; the answers feed the Phase-C pool fixture.
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
    """Phase-C pool JSONL + phase_a-shaped scaffold rows via the REAL producers.

    Returns (pool_jsonl_path, scaffold_rows). Scaffold rows come from the
    REAL stripper (`issue1345_strip_scaffolds.strip_file`) over synthetic
    stories; the pool file mirrors `issue2054_phase_c._splice_generated`'s
    realized row schema (answer carried directly, conv_id in the raw
    scaffold key space) for the SAME scaffolds. The conv_id convention is
    the exact phase_a one (`issue2054_phase_a.py` `_recover_scaffolds`:
    `row.setdefault("conv_id", row.get("scaffold_id", ...))`).
    """
    stories = tmp_path / "x_stories_1.jsonl"
    rows = [
        {"story_id": "s0001", "conv_id": "s0001", "story": STORY_1},
        {"story_id": "s0002", "conv_id": "s0002", "story": STORY_2},
    ]
    stories.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    scaffold_rows, counts = strip_cli.strip_file(stories, "Helios")
    assert counts["kept"] == 2, counts
    for i, row in enumerate(scaffold_rows):
        row.setdefault("conv_id", row.get("scaffold_id", f"char_helios_{i}"))
        row.setdefault("variant", "char_helios")
    # Pin the PRODUCTION key shape the C5 canonization exists for: the
    # stripper prefixes story_id, so a scaffold conv_id is `stripped_s...`
    # (the committed shared_fold_map.json keys have exactly this shape).
    assert scaffold_rows[0]["conv_id"] == "stripped_s0001", scaffold_rows[0]["conv_id"]

    # Phase-C on-policy pool rows for the SAME scaffolds — the realized
    # `_splice_generated` schema (conv_id raw scaffold space, answer direct).
    pool = tmp_path / "on_policy_char_helios__attrib_quoted.jsonl"
    pool_rows = [
        {
            "scaffold_id": r["scaffold_id"],
            "conv_id": r["conv_id"],
            "variant": "char_helios",
            "character": "Helios",
            "form": "attrib_quoted",
            "final_text": "",
            "answer": ans,
            "answer_start": 0,
            "answer_end": len(ans),
            "answer_len_chars": len(ans),
            "prefix_end_char": None,
        }
        for r, ans in zip(
            scaffold_rows,
            [ANSWER_1, "The core temperature is stable and the loop holds through the night."],
            strict=True,
        )
    ]
    pool.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool_rows), encoding="utf-8"
    )
    return pool, scaffold_rows


# ---------------------------------------------------------------------------
# C5 — canonical join key
# ---------------------------------------------------------------------------
def test_canon_conv_id_strips_stripper_prefix_and_is_idempotent():
    assert phase_d._canon_conv_id("stripped_s2445") == "s2445"
    assert phase_d._canon_conv_id("s2445") == "s2445"
    assert phase_d._canon_conv_id(phase_d._canon_conv_id("stripped_s2445")) == "s2445"


def test_index_pool_answers_keys_on_canonical_conv_id(tmp_path):
    pool, _ = _stripped_scaffold_rows(tmp_path)
    answers, stats = phase_d._index_pool_answers(pool, "attrib_quoted")
    assert answers["s0001"] == ANSWER_1
    assert set(answers) == {"s0001", "s0002"}
    assert stats["n_rows_seen"] == 2
    assert stats["n_form_mismatch"] == 0
    assert stats["n_empty_answer"] == 0
    assert stats["n_dup_conv"] == 0


def test_index_pool_answers_counts_anomalies_never_silent(tmp_path):
    """Form mismatch / empty answer / duplicate conv_id are counted skips."""
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"conv_id": "mt1", "form": "attrib_quoted", "answer": "A first answer."},
        {"conv_id": "mt1", "form": "attrib_quoted", "answer": "A duplicate row."},
        {"conv_id": "mt2", "form": "bare_label", "answer": "Wrong form."},
        {"conv_id": "mt3", "form": "attrib_quoted", "answer": ""},
    ]
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    answers, stats = phase_d._index_pool_answers(pool, "attrib_quoted")
    assert answers == {"mt1": "A first answer."}
    assert stats["n_dup_conv"] == 1
    assert stats["n_form_mismatch"] == 1
    assert stats["n_empty_answer"] == 1


def test_pool_path_maps_variant_tail_to_answer_model():
    """The `_op`/`_op_base` tail selects THIS task's pool model dir — never
    the parent #1345 prefix (the 2026-08-06 user directive)."""
    p_op = phase_d._pool_path_in_repo("char_vex_op", "attrib_quoted")
    p_base = phase_d._pool_path_in_repo("char_vex_op_base", "attrib_quoted")
    assert p_op == (
        "issue2054_lattice/on_policy/qwen2.5-7b-instruct/char_vex/"
        "on_policy_char_vex__attrib_quoted.jsonl"
    )
    assert p_base == (
        "issue2054_lattice/on_policy/qwen2.5-7b/char_vex/on_policy_char_vex__attrib_quoted.jsonl"
    )
    assert "issue1345" not in p_op and "issue1345" not in p_base
    with pytest.raises(ValueError):
        phase_d._answer_model_for("char_vex")  # no _op tail -> no cell (c)


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
    pool, scaffold_rows = _stripped_scaffold_rows(tmp_path)

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
    def fake_download(path_in_repo: str) -> Path:
        assert path_in_repo == (
            "issue2054_lattice/on_policy/qwen2.5-7b-instruct/char_helios/"
            "on_policy_char_helios__attrib_quoted.jsonl"
        ), path_in_repo
        return pool

    monkeypatch.setattr(phase_d, "_download_pool_file", fake_download)

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
    # Answer-source provenance (the 2026-08-06 rewire): THIS task's pool.
    assert digest["answer_source"] == "issue2054_lattice/on_policy"
    assert digest["answer_form"] == "attrib_quoted"
    per = digest["per_variant"][0]
    assert per["n_out_of_fold"] == 1
    assert per["n_no_answer"] == 0  # the join resolves for every in-fold row
    assert per["answer_model"] == "qwen2.5-7b-instruct"
    assert "issue2054_lattice/on_policy" in per["answer_pool"]
    # n-floor is REPORT-only (never a smoke-killing gate — the #1345
    # gate-calibration class): n_out=1 < 6250 flags, run still exits 0.
    assert per["floor_met"] is False
    assert per["n_floor"] == 6250
    assert digest["variants_below_floor"] == ["char_helios_op"]


def test_old_parent_pool_regime_sidecar_refuses_never_resumes(tmp_path, monkeypatch):
    """A done sidecar written under the RETIRED parent-#1345 answer source
    (regime lacking answer_source/answer_form/answer_model) must REFUSE
    (RegimeMismatch -> rc 1), never silently resume stale parent-pool
    splices — the mechanical guarantee behind the 2026-08-06 user directive
    'make sure we only use the new data'."""
    import issue2054_resume as resume

    pool, scaffold_rows = _stripped_scaffold_rows(tmp_path)
    scaff_dir = tmp_path / "scaffolds" / "char_helios"
    scaff_dir.mkdir(parents=True)
    (scaff_dir / "scaffolds_char_helios.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scaffold_rows),
        encoding="utf-8",
    )
    fold_map = tmp_path / "shared_fold_map.json"
    fold_map.write_text(json.dumps({"fold_of": {"stripped_s0001": 0}}), encoding="utf-8")

    def fake_download(path_in_repo: str) -> Path:
        return pool

    monkeypatch.setattr(phase_d, "_download_pool_file", fake_download)

    # Simulate a completed OLD-WIRING run: output + sidecar under the
    # pre-rewire regime key set (no answer_source/answer_form/answer_model).
    out_dir = tmp_path / "cell_c"
    vdir = out_dir / "char_helios_op"
    vdir.mkdir(parents=True)
    old_out = vdir / forms.phase_output_name("cell_c", "char_helios_op", "chat")
    old_out.write_text('{"conv_id": "stripped_s0001"}\n', encoding="utf-8")
    resume.write_done(
        old_out,
        regime={
            "cell": forms.cell_key("char_helios_op", "cell_c", "chat", "any"),
            "target_conv_ids": 8000,
            "fold_filter": True,
        },
        inputs={},
        extra={"n_in": 1, "n_out": 1},
    )

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
            "--target-conv-ids",
            "8000",
            "--skip-upload",
        ],
    )
    rc = phase_d.main()
    assert rc == 1  # RegimeMismatch surfaced as a loud error, not a resume
    # The stale output was NOT silently reused or overwritten.
    assert old_out.read_text(encoding="utf-8") == '{"conv_id": "stripped_s0001"}\n'


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
