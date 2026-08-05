"""Pins for the #2054 r12 cross-variant answer-conflict resolution
(``epm:failure v3``: 430 conflict events / 270 unique stripped conv_ids).

Gate-semantics change in ``scripts/issue2054_build_answers.py``:

- whitespace-only / prefix-truncation conflicts CANONICALIZE via a
  deterministic, order-independent rule (majority byte form; sha256
  tie-break; maximal normalized superstring for the prefix class) —
  fails PRE-fix: the r9 body raised ``RuntimeError`` on ANY conflict;
- substantive conflicts EXCLUDE the conv_id (manifest-persisted), and the
  hard raise SURVIVES for the beyond-tail regime
  (``substantive > max(20, 2% of the stripped union)``);
- ``scripts/issue2054_phase_b.py`` DROPS builder-excluded conv_ids (never
  the scaffold-fallback, which would re-break the cross-variant byte-fixed
  answer invariant for exactly those rows).

Boundary fakes: ``_stage_sharded_jsonl`` (the Hub staging boundary) is
monkeypatched signature-conformant to serve real tmp_path JSONLs; every
other body executes for real. Fixture strings are synthetic placeholders —
no real-corpus content, per the LMSYS digest-only discipline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_build_answers as ba  # noqa: E402
import issue2054_phase_b as phase_b  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Classification + canonicalization units


def test_classify_conflict_classes():
    assert ba._classify_conflict(["a b\nc", "a b c"]) == "whitespace_only"
    assert ba._classify_conflict(["one two", "one two three"]) == "prefix_truncation"
    assert ba._classify_conflict(["hello world", "goodbye moon"]) == "substantive"
    # Prefix must be STRICT on normalized text; ws-equal forms stay class (a).
    assert ba._classify_conflict(["x  y", "x y", "x\ny"]) == "whitespace_only"


def test_canonical_answer_majority_wins():
    per = {"v1": "a b", "v2": "a b", "v3": "a\nb"}
    assert ba._canonical_answer(per, "whitespace_only") == "a b"


def test_canonical_answer_sha_tiebreak_is_order_independent():
    forms = ["a b", "a\nb"]
    expected = min(forms, key=ba._answer_sha)
    assert ba._canonical_answer({"v1": forms[0], "v2": forms[1]}, "whitespace_only") == expected
    assert ba._canonical_answer({"v2": forms[1], "v1": forms[0]}, "whitespace_only") == expected


def test_canonical_answer_prefix_takes_maximal_superstring():
    per = {"v1": "one two", "v2": "one two three", "v3": "one two"}
    assert ba._canonical_answer(per, "prefix_truncation") == "one two three"


# ─────────────────────────────────────────────────────────────────────────────
# Resolution dispositions + digest-only audit


def test_resolve_conflicts_mixed_dispositions_and_digest_only_audit():
    collected = {
        "stripped_clean": {"v1": "same", "v2": "same"},
        "stripped_ws": {"v1": "a b", "v2": "a  b", "v3": "a b"},
        "stripped_sub": {"v1": "hello world", "v2": "goodbye moon"},
    }
    answers, excluded, tallies, audit = ba._resolve_answer_conflicts(collected)
    assert answers["stripped_clean"] == "same"
    assert answers["stripped_ws"] == "a b"  # majority byte form
    assert "stripped_sub" not in answers
    assert excluded == {"stripped_sub"}
    assert tallies == {
        "cross_variant_conflict": 2,
        "conflict_ws_canonicalized": 1,
        "conflict_prefix_canonicalized": 0,
        "conflict_substantive_excluded": 1,
    }
    # Audit rows are digest-only: sha8 + chars, NEVER raw answer text.
    assert {r["conv_id"] for r in audit} == {"stripped_ws", "stripped_sub"}
    for row in audit:
        for rec in row["per_variant"].values():
            assert set(rec) == {"sha8", "chars"}
        assert row["disposition"] in ("canonicalized", "excluded")


# ─────────────────────────────────────────────────────────────────────────────
# _scaffold_answers end-to-end (Hub staging boundary faked; real bodies)


def _fake_stage(tmp_path: Path, pools: dict[str, list[dict]], monkeypatch: pytest.MonkeyPatch):
    """Signature-conformant fake of the Hub staging boundary: serve real
    JSONLs from tmp_path; everything downstream executes the real body."""
    paths: dict[str, Path] = {}
    for v, rows in pools.items():
        p = tmp_path / f"scaffolds_{v}.jsonl"
        p.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        paths[v] = p

    def fake_stage_sharded_jsonl(dest_dir: Path, base_prefix: str, stem: str) -> Path:
        variant = stem.removeprefix("scaffolds_")
        return paths[variant]

    monkeypatch.setattr(ba, "_stage_sharded_jsonl", fake_stage_sharded_jsonl)


def test_scaffold_answers_ws_conflict_canonicalizes_instead_of_raising(
    tmp_path, monkeypatch, capsys
):
    """Fails PRE-fix: the r9 `_scaffold_answers` raised RuntimeError on any
    cross-variant byte difference; the r12 body canonicalizes class (a)."""
    pools = {
        "v1": [
            {"conv_id": "stripped_s1", "answer": "hello\nworld"},
            {"conv_id": "stripped_s2", "answer": "stable"},
        ],
        "v2": [
            {"conv_id": "stripped_s1", "answer": "hello world"},
            {"conv_id": "stripped_s2", "answer": "stable"},
        ],
    }
    _fake_stage(tmp_path, pools, monkeypatch)
    needed = {"stripped_s1", "stripped_s2"}
    answers, counters, excluded, _audit = ba._scaffold_answers(tmp_path, ["v1", "v2"], needed, None)
    assert set(answers) == needed
    assert answers["stripped_s1"] in ("hello\nworld", "hello world")
    assert counters["cross_variant_conflict"] == 1
    assert counters["conflict_ws_canonicalized"] == 1
    assert counters["conflict_substantive_excluded"] == 0
    assert counters["cross_variant_conflicts_hard"] == 0
    assert excluded == set()
    assert "conflict resolution:" in capsys.readouterr().out


def test_scaffold_answers_substantive_tail_excludes(tmp_path, monkeypatch):
    pools = {
        "v1": [{"conv_id": "stripped_s1", "answer": "alpha text"}],
        "v2": [{"conv_id": "stripped_s1", "answer": "omega prose"}],
    }
    _fake_stage(tmp_path, pools, monkeypatch)
    answers, counters, excluded, audit = ba._scaffold_answers(
        tmp_path, ["v1", "v2"], {"stripped_s1"}, None
    )
    assert answers == {}
    assert excluded == {"stripped_s1"}
    assert counters["conflict_substantive_excluded"] == 1
    assert counters["cross_variant_conflicts_hard"] == 0
    assert audit[0]["disposition"] == "excluded"


def test_scaffold_answers_hard_cap_raises_on_systemic_divergence(tmp_path, monkeypatch):
    """Beyond-tail regime: substantive exclusions past max(20, 2% of the
    stripped union) keep the ORIGINAL fail-loud contract."""
    n = 25  # union 25 -> cap = max(20, ceil(0.5)) = 20; 25 > 20 raises
    pools = {
        "v1": [{"conv_id": f"stripped_s{i}", "answer": f"alpha {i}"} for i in range(n)],
        "v2": [{"conv_id": f"stripped_s{i}", "answer": f"omega {i}"} for i in range(n)],
    }
    _fake_stage(tmp_path, pools, monkeypatch)
    needed = {f"stripped_s{i}" for i in range(n)}
    with pytest.raises(RuntimeError, match="cross_variant_conflicts_hard=25"):
        ba._scaffold_answers(tmp_path, ["v1", "v2"], needed, None)


def test_collect_kept_cap_caps_new_cids_but_keeps_cross_variant_additions():
    counters = {
        "scaffold_rows_read": 0,
        "missing_answer_field": 0,
        "sentinel_in_answer": 0,
        "intra_variant_duplicate": 0,
    }
    collected: dict[str, dict[str, str]] = {}
    needed = {"stripped_s1", "stripped_s2"}
    ba._collect_scaffold_answers(
        [{"conv_id": "stripped_s1", "answer": "a"}], "v1", needed, 1, counters, collected
    )
    # Cap reached: a NEW cid is refused ...
    ba._collect_scaffold_answers(
        [{"conv_id": "stripped_s2", "answer": "b"}], "v1", needed, 1, counters, collected
    )
    # ... but another VARIANT of an already-collected cid still lands (so the
    # smoke slice exercises cross-variant resolution).
    ba._collect_scaffold_answers(
        [{"conv_id": "stripped_s1", "answer": "a2"}], "v2", needed, 1, counters, collected
    )
    assert set(collected) == {"stripped_s1"}
    assert collected["stripped_s1"] == {"v1": "a", "v2": "a2"}
    # Intra-variant duplicate: counted, first kept.
    ba._collect_scaffold_answers(
        [{"conv_id": "stripped_s1", "answer": "dupe"}], "v1", needed, 1, counters, collected
    )
    assert counters["intra_variant_duplicate"] == 1
    assert collected["stripped_s1"]["v1"] == "a"


# ─────────────────────────────────────────────────────────────────────────────
# phase_b: builder-excluded conv_ids are DROPPED (never scaffold-fallback)


def _scaffold_row(i: int) -> dict:
    return {
        "scaffold_id": f"stripped_s{i}",
        "conv_id": f"stripped_s{i}",
        "character": "Helios",
        "scaffold_text": f"A scene about question {i}.",
        "question": f"Question number {i}?",
        "answer": f"scaffold-original answer {i}",
    }


def test_phase_b_excluded_conv_id_dropped_even_with_pool_hit(tmp_path):
    scaffolds = tmp_path / "scaffolds.jsonl"
    with scaffolds.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps(_scaffold_row(i)) + "\n")
    # s1 is excluded AND has a pool answer: exclusion must win (drop, no splice).
    answers = {"stripped_s0": "pool answer zero", "stripped_s1": "pool answer one"}
    counts, out_path = phase_b._process_variant(
        "char_helios",
        scaffolds,
        answers,
        tmp_path / "out",
        "chat",
        excluded={"stripped_s1"},
    )
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [r["conv_id"] for r in rows] == ["stripped_s0", "stripped_s2"]
    assert counts["n_excluded_conflict"] == 1
    assert counts["n_answer_from_pool"] == 1
    assert counts["n_answer_from_scaffold_fallback"] == 1


def test_phase_b_load_excluded_manifest(tmp_path):
    answers_path = tmp_path / "answers_pool.jsonl"
    answers_path.write_text("", encoding="utf-8")
    # Absent manifest -> empty set (pre-r12 pools).
    assert phase_b._load_excluded_conv_ids(answers_path) == set()
    (tmp_path / "answers_excluded_conv_ids.json").write_text(
        json.dumps({"excluded": ["stripped_s7", "stripped_s9"]}), encoding="utf-8"
    )
    assert phase_b._load_excluded_conv_ids(answers_path) == {"stripped_s7", "stripped_s9"}


def test_phase_b_missing_manifest_hard_fails_when_meta_declares_exclusions(tmp_path):
    """r13 fail-closed pin: an r12+ pool whose meta declares substantive
    exclusions MUST NOT load with an absent exclusion manifest — a consumer
    staging the pool without the sidecar would silently re-splice the
    excluded conv_ids via scaffold-fallback (the 2x2 invariant r12 protects).
    The raise names BOTH files (meta + expected manifest)."""
    answers_path = tmp_path / "answers_pool.jsonl"
    answers_path.write_text("", encoding="utf-8")
    (tmp_path / "answers_pool.meta.json").write_text(
        json.dumps({"conflict_resolution": {"substantive_excluded": 15}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError) as ei:
        phase_b._load_excluded_conv_ids(answers_path)
    assert "answers_pool.meta.json" in str(ei.value)
    assert "answers_excluded_conv_ids.json" in str(ei.value)
    # Manifest present -> loads normally even with meta declaring exclusions.
    (tmp_path / "answers_excluded_conv_ids.json").write_text(
        json.dumps({"excluded": ["stripped_s7"]}), encoding="utf-8"
    )
    assert phase_b._load_excluded_conv_ids(answers_path) == {"stripped_s7"}


def test_phase_b_missing_manifest_permissive_when_meta_declares_zero(tmp_path):
    """Meta present but zero substantive exclusions -> absent sidecar stays
    legal (same as the no-meta pre-r12 case pinned above)."""
    answers_path = tmp_path / "answers_pool.jsonl"
    answers_path.write_text("", encoding="utf-8")
    (tmp_path / "answers_pool.meta.json").write_text(
        json.dumps({"conflict_resolution": {"substantive_excluded": 0}}),
        encoding="utf-8",
    )
    assert phase_b._load_excluded_conv_ids(answers_path) == set()
