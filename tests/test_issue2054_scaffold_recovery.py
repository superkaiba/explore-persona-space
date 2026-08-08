"""Pins for the #2054 crash-fix round 5 (phase-A supply shortfall).

Three defects, measured on the 2026-08-05 production `--stage gen` run and
fixed together:

R5-1  `issue1345_scaffold_common.attrib_re_for` built a case-SENSITIVE
      speaker pattern, so a variant whose parent stories label the speaker
      ALL-CAPS ("HELIOS replied:") parsed ZERO turns and `strip_story`
      rejected them all as `no_parsed_turns`. Measured against the real
      parent files: char_helios recovered 12/2,187 pre-fix, 2,179/2,187
      post-fix (char_wren/dana/vex/assistant unchanged, title-case).

R5-2  the shared question draw admitted MULTILINE and very long questions
      that the plan-req-6 verbatim filter then rejected downstream — after
      their generation compute was already spent (min per-variant
      verbatim-keep 26.9% overall vs 36.0% under single-line + <400 chars).

R5-3  `n_gen` was sized off the CROSS-variant recovered intersection while
      kill gate 4 intersects WITHIN one (character, model) group, so the
      default under-draws whenever per-variant recovery exceeds the shared
      intersection. `--gen-draw-n` sizes the shared draw directly.

Fixtures are synthetic prose written for this test — no real-corpus text.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_common as c  # noqa: E402
import issue1345_scaffold_common as sc  # noqa: E402
import issue2054_phase_a as pa  # noqa: E402

# ---------------------------------------------------------------------------
# R5-1 — attrib_re_for matches the character name case-insensitively
# ---------------------------------------------------------------------------


def _story_with_speaker(label: str) -> str:
    """A minimal two-turn story rendered under the parent's own template,
    with the assistant speaker labelled `label`."""
    base = c.ANSWER_ATTRIB_RE.pattern
    assert c.STORY_CHARACTER_NAME in base or "ARIA" in base
    return (
        "The workshop was quiet.\n\n"
        'The visitor asked, "How do I keep a sourdough starter alive?"\n\n'
        f'{label} replied, "Feed it flour and water on a steady rhythm."\n\n'
        "Outside, the rain kept its own time.\n"
    )


def test_attrib_re_for_matches_all_caps_speaker_label():
    """FAILS PRE-FIX: the case-sensitive pattern found zero turns for an
    ALL-CAPS speaker label, which is exactly what emptied char_helios."""
    turns = sc.parse_story_turns_for(_story_with_speaker("HELIOS"), "Helios")
    assert turns, "ALL-CAPS speaker label must still parse as an attributed turn"


def test_attrib_re_for_matches_title_case_speaker_label():
    """Non-regression: the title-case label the other variants use still
    parses (it did pre-fix too)."""
    turns = sc.parse_story_turns_for(_story_with_speaker("Helios"), "Helios")
    assert turns


def test_attrib_re_for_does_not_match_a_different_name():
    """The relaxation is CASING only — a different character must not match,
    or the strip would attribute another speaker's turns to this variant."""
    turns = sc.parse_story_turns_for(_story_with_speaker("Wren"), "Helios")
    assert not turns


def test_attrib_re_for_does_not_match_the_lowercase_english_word():
    """The alternation is {given, ALL-CAPS} — deliberately NOT a blanket
    `(?i:...)`. Several lattice names are ordinary English words, and a
    fully case-insensitive token would attribute a narrator clause to the
    character (silent mis-slicing, not a visible kept-count drop)."""
    narrator = 'Only to vex her, the courier said, "The road is closed."'
    assert not sc.attrib_re_for("Vex").search(narrator)
    assert not sc.attrib_re_for("Wren").search('a wren sang; she said, "Now."')
    # ...while both observed casings of the real speaker still match.
    assert sc.attrib_re_for("Vex").search('Vex said, "The road is closed."')
    assert sc.attrib_re_for("Vex").search('VEX said, "The road is closed."')


def test_attrib_re_for_keeps_group_numbering():
    """`strip_story` re-aligns the opening quote on `m.end(1)`, so the
    case-insensitive wrapper must stay NON-capturing — a capturing group
    would shift every group index by one and silently mis-slice each turn."""
    swapped = sc.attrib_re_for("Helios")
    assert swapped.groups == c.ANSWER_ATTRIB_RE.groups
    m = swapped.search(_story_with_speaker("HELIOS"))
    assert m is not None
    # group(1) must still be the quote-opening group the caller re-aligns on.
    assert m.group(1) == '"'


# ---------------------------------------------------------------------------
# R5-2 — the question-pool admission filter
# ---------------------------------------------------------------------------


def test_question_max_chars_is_scaffold_admissible():
    """FAILS PRE-FIX (was 8000): >=800-char questions verbatim-keep at
    1.1-2.8%, so admitting them burns generation compute on rows the
    downstream req-6 filter rejects."""
    assert pa.QUESTION_MAX_CHARS == 400


def test_question_single_line_filter_is_on():
    """FAILS PRE-FIX (constant did not exist): multiline questions
    verbatim-keep at 1.3-3.0%."""
    assert pa.QUESTION_SINGLE_LINE is True


def test_question_pool_fingerprint_covers_the_new_filters():
    """A cached draw built under the OLD bounds must not be silently reused:
    both admission knobs are output-affecting, so both enter the fingerprint."""
    base = pa._question_pool_fingerprint(100, 137, "rev")
    orig_max, orig_single = pa.QUESTION_MAX_CHARS, pa.QUESTION_SINGLE_LINE
    try:
        pa.QUESTION_MAX_CHARS = 8000
        assert pa._question_pool_fingerprint(100, 137, "rev") != base
    finally:
        pa.QUESTION_MAX_CHARS = orig_max
        pa.QUESTION_SINGLE_LINE = orig_single


# ---------------------------------------------------------------------------
# R5-3 — --gen-draw-n sizes the shared draw directly
# ---------------------------------------------------------------------------


def test_gen_draw_n_is_in_the_gen_resume_regime():
    """#722-r3: every output-affecting arg is part of the resume key, or a
    resumed pool built under a different draw size is silently reused."""

    class _Args:
        seed = 137
        target_conv_ids = 8000
        gen_draw_n = 12000
        gen_model = "instruct"
        gen_mock = False
        no_generate = False

    regime = pa._gen_regime("char_helios", _Args())
    assert regime["gen_draw_n"] == 12000

    _Args.gen_draw_n = None
    assert pa._gen_regime("char_helios", _Args())["gen_draw_n"] is None


# ---------------------------------------------------------------------------
# R6 — --prejudge-from-hf must reassemble the SHARDED upload form
# ---------------------------------------------------------------------------


def _fake_hub(monkeypatch, remote: dict[str, bytes]):
    """Signature-conformant fakes at the HF network boundary only.

    `remote` maps path_in_repo -> bytes. Returns the list of (path, overwrite)
    staging calls so a test can assert re-staging actually happened.
    """
    import explore_persona_space.orchestrate.hub as hub

    calls: list[tuple[str, bool]] = []

    def fake_stage_hub_file(
        repo_id, path_in_repo, target, *, repo_type="dataset",
        revision=None, token=None, overwrite=False,
    ):
        from pathlib import Path as _P

        calls.append((path_in_repo, overwrite))
        target = _P(target)
        if target.exists() and not overwrite:
            return target
        if path_in_repo not in remote:
            raise FileNotFoundError(path_in_repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(remote[path_in_repo])
        return target

    def fake_retry_transient(fn, *, what=None, **kw):
        return fn()

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage_hub_file)
    monkeypatch.setattr(hub, "retry_transient", fake_retry_transient)

    class _FakeApi:
        def file_exists(self, repo_id, filename, *, repo_type="dataset", **kw):
            return filename in remote

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _FakeApi())
    return calls


def _sharded_remote(variant: str, rows: list[bytes]) -> tuple[dict[str, bytes], bytes]:
    import hashlib as _h
    import json as _j

    stem = f"scaffolds_{variant}_prejudge"
    base = f"issue2054_lattice/scaffolds/{variant}"
    mid = len(rows) // 2
    p0, p1 = b"".join(rows[:mid]), b"".join(rows[mid:])
    names = [f"{stem}.shard00.jsonl", f"{stem}.shard01.jsonl"]
    remote = {
        f"{base}/{names[0]}": p0,
        f"{base}/{names[1]}": p1,
        f"{base}/{stem}.manifest.json": _j.dumps(
            {
                "source": f"{stem}.jsonl",
                "parts": names,
                "line_counts": [mid, len(rows) - mid],
                "sha256": {
                    names[0]: _h.sha256(p0).hexdigest(),
                    names[1]: _h.sha256(p1).hexdigest(),
                },
            }
        ).encode(),
        f"{base}/{stem}.jsonl.done.json": b'{"regime": {}, "extra": {}}',
        # PRIOR-ROUND RESIDUE at the plain name — smaller, and what the r6
        # defect staged instead of the shards.
        f"{base}/{stem}.jsonl": b'{"conv_id": "STALE"}\n',
    }
    return remote, p0 + p1


def test_prejudge_staging_reassembles_shards_byte_exactly(tmp_path, monkeypatch):
    """FAILS PRE-FIX: the old staging pulled `<stem>.jsonl` — on HF that name
    is the previous round's UNSHARDED residue, so the judge leg silently
    consumed the failed round's pool."""
    rows = [b'{"conv_id": "c%03d"}\n' % i for i in range(10)]
    remote, expected = _sharded_remote("char_helios", rows)
    _fake_hub(monkeypatch, remote)

    pa._stage_prejudge_from_hf(tmp_path, "char_helios")

    got = pa._prejudge_path(tmp_path, "char_helios").read_bytes()
    assert got == expected, "reassembled pool must be byte-identical to the shard concat"
    assert b"STALE" not in got


def test_prejudge_staging_falls_back_to_unsharded_when_no_manifest(tmp_path, monkeypatch):
    base = "issue2054_lattice/scaffolds/char_wren"
    stem = "scaffolds_char_wren_prejudge"
    body = b'{"conv_id": "only"}\n'
    _fake_hub(
        monkeypatch,
        {
            f"{base}/{stem}.jsonl": body,
            f"{base}/{stem}.jsonl.done.json": b"{}",
        },
    )
    pa._stage_prejudge_from_hf(tmp_path, "char_wren")
    assert pa._prejudge_path(tmp_path, "char_wren").read_bytes() == body


def test_prejudge_staging_overwrites_a_stale_local_pool(tmp_path, monkeypatch):
    """`stage_hub_file` is idempotent (existing target -> no network call), so
    without overwrite=True a stale local pool wins over the Hub copy."""
    rows = [b'{"conv_id": "c%03d"}\n' % i for i in range(6)]
    remote, expected = _sharded_remote("char_dana", rows)
    calls = _fake_hub(monkeypatch, remote)

    stale = pa._prejudge_path(tmp_path, "char_dana")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b'{"conv_id": "PRIOR_ROUND"}\n')

    pa._stage_prejudge_from_hf(tmp_path, "char_dana")

    assert stale.read_bytes() == expected
    assert all(ow for _, ow in calls), "every staging call must force overwrite"


def test_load_prejudge_restages_every_variant_not_just_missing(tmp_path, monkeypatch):
    """The stale-local trap: `missing` was empty because prior-round files sat
    on disk, so the old code staged NOTHING and the staleness gate could only
    refuse."""
    staged: list[str] = []
    monkeypatch.setattr(
        pa, "_stage_prejudge_from_hf", lambda out_dir, v: staged.append(v)
    )
    monkeypatch.setattr(pa, "_verify_prejudge_staleness", lambda *a, **k: None)
    monkeypatch.setattr(pa, "_read_jsonl", lambda p: [])

    variants = ["char_helios", "char_wren"]
    for v in variants:  # both already present locally = the trap
        p = pa._prejudge_path(tmp_path, v)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n", encoding="utf-8")

    pa._load_prejudge(tmp_path, variants, from_hf=True, args=None)
    assert staged == variants, "from_hf must re-stage EVERY variant, not only missing ones"
