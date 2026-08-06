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


# ---------------------------------------------------------------------------
# r15 pins: shared_question_draw staging is MANIFEST-FIRST (epm:failure v4 —
# the top-up gen leg uploads the >9.5 MB draw SHARDED only, so the plain hub
# name is stale prior-round residue; the loader must prefer the sharded form
# and fall back to the unsharded name ONLY when no manifest exists on HF).
# These fail pre-fix: `_stage_draw_jsonl` did not exist and `_draw_questions`
# staged the unsharded name unconditionally.
# Boundary fakes only: `_api().file_exists` + `stage_hub_file` (network) and
# `_stage_sharded_jsonl` (the Hub staging boundary, per the established
# test_issue2054_answer_conflicts.py convention); the real sharded body runs
# in the production rebuild.
# ---------------------------------------------------------------------------
import hashlib  # noqa: E402

import issue2054_build_answers as ba  # noqa: E402
import issue2054_verify_scaffold_uploads as vsu  # noqa: E402


class _FakeHfApi:
    def __init__(self, manifest_exists: bool):
        self._exists = manifest_exists
        self.probed: list[str] = []

    def file_exists(self, repo_id, path_in_repo, repo_type=None):
        self.probed.append(path_in_repo)
        return self._exists


def test_stage_draw_prefers_sharded_form_when_manifest_present(tmp_path, monkeypatch):
    import explore_persona_space.orchestrate.hub as hub

    api = _FakeHfApi(manifest_exists=True)
    monkeypatch.setattr(ba, "_api", lambda: api)
    calls: dict[str, tuple] = {}

    def fake_sharded(dest_dir: Path, base_prefix: str, stem: str) -> Path:
        calls["sharded"] = (dest_dir, base_prefix, stem)
        p = dest_dir / f"{stem}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
        return p

    monkeypatch.setattr(ba, "_stage_sharded_jsonl", fake_sharded)

    def refuse_unsharded(*a, **k):
        raise AssertionError("unsharded fallback must NOT fire when a manifest exists")

    monkeypatch.setattr(hub, "stage_hub_file", refuse_unsharded)
    out = ba._stage_draw_jsonl(tmp_path)
    assert calls["sharded"] == (
        tmp_path / ba.SCAFFOLDS_PREFIX,
        ba.SCAFFOLDS_PREFIX,
        "shared_question_draw",
    )
    assert out.name == "shared_question_draw.jsonl"
    assert api.probed == [f"{ba.SCAFFOLDS_PREFIX}/shared_question_draw.manifest.json"]


def test_stage_draw_falls_back_only_when_no_manifest(tmp_path, monkeypatch):
    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(ba, "_api", lambda: _FakeHfApi(manifest_exists=False))

    def refuse_sharded(*a, **k):
        raise AssertionError("sharded stager must NOT run when no manifest exists")

    monkeypatch.setattr(ba, "_stage_sharded_jsonl", refuse_sharded)
    staged: dict[str, object] = {}

    def fake_stage_hub_file(repo_id, path_in_repo, target, *, repo_type, overwrite=False, **k):
        staged["path_in_repo"] = path_in_repo
        staged["overwrite"] = overwrite
        t = Path(target)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text("", encoding="utf-8")
        return t

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage_hub_file)
    out = ba._stage_draw_jsonl(tmp_path)
    assert staged["path_in_repo"] == f"{ba.SCAFFOLDS_PREFIX}/shared_question_draw.jsonl"
    assert staged["overwrite"] is True  # authoritative overwrite — never a stale local reuse
    assert out == tmp_path / ba.SCAFFOLDS_PREFIX / "shared_question_draw.jsonl"


def _fake_vsu_stage(manifest: dict, shard_bytes: bytes):
    def fake_stage(api, path_in_repo: str, dest: Path):
        d = Path(dest)
        d.parent.mkdir(parents=True, exist_ok=True)
        if path_in_repo.endswith(".manifest.json"):
            d.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            d.write_bytes(shard_bytes)
        return d

    return fake_stage


def test_verify_shared_draw_is_manifest_first_and_ignores_stale_unsharded(tmp_path, monkeypatch):
    shard = b'{"conv_id": "mt_a", "question": "q"}\n'
    man = {
        "source": "shared_question_draw.jsonl",
        "parts": ["shared_question_draw.shard00.jsonl"],
        "line_counts": [1],
        "sha256": {"shared_question_draw.shard00.jsonl": hashlib.sha256(shard).hexdigest()},
    }
    monkeypatch.setattr(vsu, "_stage", _fake_vsu_stage(man, shard))
    # Stale unsharded name ALSO present — must be ignored in favour of the manifest.
    listing = {
        f"{vsu.PREFIX}/shared_question_draw.manifest.json",
        f"{vsu.PREFIX}/shared_question_draw.meta.json",
        f"{vsu.PREFIX}/shared_question_draw.shard00.jsonl",
        f"{vsu.PREFIX}/shared_question_draw.jsonl",
    }
    ok, problems, n_rows = vsu.verify_shared_draw(None, listing, tmp_path, expect=1)
    assert ok and not problems and n_rows == 1
    # Row-count reconciliation binds against the MANIFEST count, so a stale
    # expectation fails even with every shard present + sha-clean.
    ok2, problems2, _ = vsu.verify_shared_draw(None, listing, tmp_path, expect=2)
    assert not ok2 and any("!= expected 2" in p for p in problems2)


def test_verify_shared_draw_flags_missing_shard_and_unsharded_fallback(tmp_path, monkeypatch):
    shard = b'{"conv_id": "mt_a", "question": "q"}\n'
    man = {
        "source": "shared_question_draw.jsonl",
        "parts": ["shared_question_draw.shard00.jsonl", "shared_question_draw.shard01.jsonl"],
        "line_counts": [1, 1],
        "sha256": {"shared_question_draw.shard00.jsonl": hashlib.sha256(shard).hexdigest()},
    }
    monkeypatch.setattr(vsu, "_stage", _fake_vsu_stage(man, shard))
    listing = {
        f"{vsu.PREFIX}/shared_question_draw.manifest.json",
        f"{vsu.PREFIX}/shared_question_draw.meta.json",
        f"{vsu.PREFIX}/shared_question_draw.shard00.jsonl",
    }
    ok, problems, _ = vsu.verify_shared_draw(None, listing, tmp_path, expect=None)
    assert not ok and any("shard01" in p and "NOT on HF" in p for p in problems)
    # No manifest at all -> pre-shard compat: unsharded accepted with the NOTE.
    listing2 = {
        f"{vsu.PREFIX}/shared_question_draw.meta.json",
        f"{vsu.PREFIX}/shared_question_draw.jsonl",
    }
    ok2, problems2, n_rows2 = vsu.verify_shared_draw(None, listing2, tmp_path, expect=None)
    assert n_rows2 == 1
    assert not ok2 and any("NOTE unsharded form only" in p for p in problems2)
