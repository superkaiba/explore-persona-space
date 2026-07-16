"""#1345 assistant-named-story name seam (plan v6 §4).

Pins three things:

1. **Default byte-identity.** With the env seam unset, ``STORY_SYSTEM_PROMPT``,
   ``JUDGE_SYSTEM``, and ``ANSWER_ATTRIB_RE.pattern`` are BYTE-IDENTICAL to the
   parent ARIA literals — the gen/bundle fingerprints hash these strings, so a
   drift here silently invalidates the parent's content-keyed resume — and the
   regex matches the ARIA form only.
2. **Env-seam threading.** Under ``EPM_STORY_CHARACTER_NAME=Assistant`` +
   ``EPM_I1345_VARIANT=assistant_named_story``, the attribution regex matches
   ``Assistant replied:`` (and NOT the ARIA form), the story/judge prompts and
   the tier-2 user message carry the new name, and every output dir + HF
   prefix is variant-scoped (never clobber the parent).
3. **Fail-loud pairing.** A non-default name WITHOUT a variant refuses at
   module import (RuntimeError) AND at the dispatch arg parser (bash exit!=0).

Env permutations run in SUBPROCESSES: the module computes the constants at
import time, and a partially-executed ``importlib.reload`` after the fail-loud
raise would corrupt the in-process module namespace for sibling tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEAM_KEYS = ("EPM_STORY_CHARACTER_NAME", "EPM_I1345_VARIANT")

# Frozen parent literals (the #1345 ARIA run @ b7ab0b3b75) — never rebuild these
# from the module under test.
PARENT_STORY_SYSTEM_PROMPT = (
    "You are writing a short story in which an AI assistant named ARIA is a "
    "character. In the story, a person asks ARIA questions and ARIA answers them "
    "helpfully and accurately. Write a realistic narrative scene of 300-500 words "
    "with 4-6 question-answer exchanges embedded naturally in the prose. Write "
    "flowing narrative prose (never a script or 'Name:' dialogue format). Put "
    "spoken dialogue in double quotes, and introduce each of ARIA's answers with "
    'an attribution BEFORE the quotation (for example: ARIA replied: "...").'
)
PARENT_JUDGE_SYSTEM = (
    "You evaluate whether a short story contains extractable question-answer "
    "dialogue. A story PASSES when it contains at least "
    "4 DISTINCT question->answer exchanges in which a person "
    "asks a question (quoted dialogue) and an AI assistant character named ARIA "
    "gives a substantive answer in quoted dialogue introduced by an attribution "
    'before the quote (e.g. ARIA replied: "..."). Exchanges with trivial or '
    "empty answers, unquoted dialogue, or answers not attributed to ARIA do not "
    "count. First give 2-3 sentences of reasoning, then finish with EXACTLY two "
    "final lines:\nTURNS: <integer number of qualifying exchanges>\n"
    "VERDICT: PASS or FAIL"
)
PARENT_ATTRIB_PATTERN = (
    r"\bARIA\b[^\"“”\n]{0,40}?"
    r"(?:said|replied|answered|responded|explained|noted|added|confirmed|"
    r"clarified|continued)"
    r"[^\"“”\n]{0,20}?([\"“])"
)


def _clean_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SEAM_KEYS}
    env.update(overrides or {})
    return env


def _run_py(code: str, overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_clean_env(overrides),
        cwd=_REPO_ROOT,
        timeout=300,
    )


def test_default_mode_byte_identity_and_aria_only_regex(monkeypatch):
    """(1) Default constants byte-equal the parent literals; regex is ARIA-form."""
    for k in _SEAM_KEYS:
        monkeypatch.delenv(k, raising=False)
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import issue1345_common as c
    import issue1345_gen_stories as g

    assert c.STORY_CHARACTER_NAME == "ARIA"
    assert c.STORY_SYSTEM_PROMPT == PARENT_STORY_SYSTEM_PROMPT
    assert g.JUDGE_SYSTEM == PARENT_JUDGE_SYSTEM
    assert c.ANSWER_ATTRIB_RE.pattern == PARENT_ATTRIB_PATTERN
    assert c.ANSWER_ATTRIB_RE.search('ARIA replied: "Certainly."')
    assert not c.ANSWER_ATTRIB_RE.search('Assistant replied: "Certainly."')
    # Default (non-variant) dirs + prefixes — the parent layout, untouched.
    assert c.DATA_DIR.as_posix() == "data/issue_1345"
    assert c.EVAL_DIR.as_posix() == "eval_results/issue_1345"
    assert c.HF_ISSUE_PREFIX == "issue1345_framing"
    assert c.HF_SMOKE_PREFIX == "issue1345_smoke"


def test_env_seam_threads_assistant_name_and_variant_scoping():
    """(2) Assistant-named seam: regex/prompts/user-msg + variant dirs/prefixes."""
    code = """
import sys
sys.path.insert(0, "scripts")
import issue1345_common as c
import issue1345_gen_stories as g

assert c.STORY_CHARACTER_NAME == "Assistant", c.STORY_CHARACTER_NAME
assert c.ANSWER_ATTRIB_RE.search('Assistant replied: "Of course."')
assert c.ANSWER_ATTRIB_RE.search('Assistant said, "Yes."')
assert not c.ANSWER_ATTRIB_RE.search('ARIA replied: "Of course."')
# Case-sensitive \\b-bounded name (parent semantics): lowercase never matches.
assert not c.ANSWER_ATTRIB_RE.search('the assistant replied: "Of course."')
assert "named Assistant is a" in c.STORY_SYSTEM_PROMPT
assert "ARIA" not in c.STORY_SYSTEM_PROMPT
assert "ARIA" not in g.JUDGE_SYSTEM and "named Assistant" in g.JUDGE_SYSTEM
msg = g.build_prompt("What is a star?", "pretrained", None)  # tokenizer unused (base branch)
assert "questions to Assistant" in msg and "ARIA" not in msg
# Variant scoping: every output dir + HF prefix lands one level deeper.
assert c.DATA_DIR.as_posix() == "data/issue_1345/assistant_named_story", c.DATA_DIR
assert c.EVAL_DIR.as_posix() == "eval_results/issue_1345/assistant_named_story"
assert c.FIG_DIR.as_posix() == "figures/issue_1345/assistant_named_story"
assert c.HF_ISSUE_PREFIX == "issue1345_framing/assistant_named_story"
assert c.HF_SMOKE_PREFIX == "issue1345_smoke/assistant_named_story"
# The REUSE pins name the PARENT's own upload — never variant-scoped.
assert c.REUSE_TENSOR_PREFIX == "issue1345_framing/analysis_tensors/turnstore"
print("SEAM-OK")
"""
    proc = _run_py(
        code,
        {
            "EPM_STORY_CHARACTER_NAME": "Assistant",
            "EPM_I1345_VARIANT": "assistant_named_story",
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SEAM-OK" in proc.stdout


def test_nondefault_name_without_variant_fails_loud_at_import():
    """(3a) Module import refuses a non-default name with no variant."""
    proc = _run_py(
        "import sys; sys.path.insert(0, 'scripts'); import issue1345_common",
        {"EPM_STORY_CHARACTER_NAME": "Assistant"},
    )
    assert proc.returncode != 0
    assert "EPM_I1345_VARIANT" in proc.stderr and "clobber" in proc.stderr


def test_dispatch_flag_pairing_fails_loud():
    """(3b) Dispatch arg parser refuses --character-name without --variant."""
    proc = subprocess.run(
        ["bash", "scripts/issue1345_dispatch.sh", "--character-name", "Assistant", "--dry-run"],
        capture_output=True,
        text=True,
        env=_clean_env({"REPO_ROOT": str(_REPO_ROOT)}),
        cwd=_REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "--variant" in proc.stderr


def _import_fit_cells(monkeypatch):
    for k in _SEAM_KEYS:
        monkeypatch.delenv(k, raising=False)
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import issue1345_fit_cells as fcells

    return fcells


def _write_cell_json(d: Path, cid: str, l19: float) -> None:
    vals = [0.0] * 28
    vals[19] = l19
    (d / f"cells_{cid}.json").write_text(__import__("json").dumps({"r2_per_layer_obs": vals}))


def test_refit_equality_halts_on_production_miss(monkeypatch, tmp_path):
    """Production refit-equality (plan v6 §7): >1e-3 drift exits 3; smoke is
    informational; matching values pass. Real body — the smoke leg can never
    reach the HALT branch (gate-calibration rule iv: unit-pin it)."""
    import pytest

    fcells = _import_fit_cells(monkeypatch)
    out_dir, ref_dir = tmp_path / "out", tmp_path / "ref"
    out_dir.mkdir()
    ref_dir.mkdir()
    cells = [{"cell_id": "R_instruct_r1_context", "regime": "r1"}]
    _write_cell_json(ref_dir, "R_instruct_r1_context", 0.6542)
    # (a) match within tol -> PASS, no raise, verdict JSON written
    _write_cell_json(out_dir, "R_instruct_r1_context", 0.6542 + 5e-4)
    fcells.refit_equality_check(out_dir, ref_dir, cells, smoke=False)
    verdict = __import__("json").loads((out_dir / "refit_equality.json").read_text())
    assert verdict["pass"] is True
    # (b) drift > tol -> HALT exit 3
    _write_cell_json(out_dir, "R_instruct_r1_context", 0.6542 + 5e-3)
    with pytest.raises(SystemExit) as ei:
        fcells.refit_equality_check(out_dir, ref_dir, cells, smoke=False)
    assert ei.value.code == 3
    # (c) same drift under smoke -> informational, no raise, pass is None
    fcells.refit_equality_check(out_dir, ref_dir, cells, smoke=True)
    verdict = __import__("json").loads((out_dir / "refit_equality.json").read_text())
    assert verdict["pass"] is None
    # (d) r3 cells are never checked (new corpus by design)
    fcells.refit_equality_check(
        out_dir, ref_dir, [{"cell_id": "R_instruct_r3_context", "regime": "r3"}], smoke=False
    )


def test_build_matched_staged_parent_allowlist_equality(monkeypatch, tmp_path):
    """Production staged-allowlist equality: a shrunken staged turnstore fails
    loud; an equal one passes; smoke demotes to a subset check. Real
    build_matched body over minimal shard sidecars."""
    import json as _json

    import pytest

    fcells = _import_fit_cells(monkeypatch)
    import issue1345_common as c

    ts, md = tmp_path / "ts", tmp_path / "md"
    ts.mkdir()
    md.mkdir()
    convs = [f"s{i}" for i in range(6)]
    for model in c.MODELS:
        for regime in ("r1", "r2"):
            (ts / f"{c.stem_for(model, regime)}_shard0.json").write_text(
                _json.dumps({"conv_ids": convs})
            )
    # (a) parent allowlist == rebuilt intersection -> PASS
    (md / "matched_subsets_parent.json").write_text(_json.dumps({"shared_r1r2_convs": convs}))
    out = fcells.build_matched(ts, md, r3_models=set(), smoke=False)
    assert out["shared_r1r2_convs"] == sorted(convs)
    # (b) parent allowlist SUPERSET (incomplete staging) -> RuntimeError
    (md / "matched_subsets_parent.json").write_text(
        _json.dumps({"shared_r1r2_convs": [*convs, "s99"]})
    )
    with pytest.raises(RuntimeError, match="staged turnstore incomplete"):
        fcells.build_matched(ts, md, r3_models=set(), smoke=False)
    # (c) same superset under smoke -> subset check passes (shard000-only grain)
    fcells.build_matched(ts, md, r3_models=set(), smoke=True)
    # (d) FOREIGN convs in the rebuilt set fail loud even under smoke
    (md / "matched_subsets_parent.json").write_text(_json.dumps({"shared_r1r2_convs": convs[:3]}))
    with pytest.raises(RuntimeError, match="NOT a subset"):
        fcells.build_matched(ts, md, r3_models=set(), smoke=True)


def test_extract_name_assert_mismatch_and_default(monkeypatch, tmp_path):
    """Extract-entry env-mismatch guard (plan v6 §4): a stored name differing
    from the runtime constant fails loud; matching / parent-era (field-less)
    yield JSONs pass under the ARIA default. Real body (the GPU-bound extract
    main() is smoke-carved-out; this pins the guard directly)."""
    import json as _json

    import pytest

    for k in _SEAM_KEYS:
        monkeypatch.delenv(k, raising=False)
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import issue1345_extract_turnstore as ext

    yp = tmp_path / "story_yield_instruct.json"
    # (a) Assistant-era yield vs ARIA-default runtime -> AssertionError
    yp.write_text(_json.dumps({"story_character_name": "Assistant"}))
    with pytest.raises(AssertionError, match="name mismatch"):
        ext.assert_story_character_name(tmp_path, "instruct")
    # (b) matching name -> pass
    yp.write_text(_json.dumps({"story_character_name": "ARIA"}))
    ext.assert_story_character_name(tmp_path, "instruct")
    # (c) parent-era yield JSON (field absent) -> ARIA default, pass
    yp.write_text(_json.dumps({"model": "instruct"}))
    ext.assert_story_character_name(tmp_path, "instruct")
    # (d) missing yield report -> fail loud
    with pytest.raises(AssertionError, match="missing"):
        ext.assert_story_character_name(tmp_path, "pretrained")


def test_dispatch_rejects_unsafe_slugs():
    """Charset guards: names/slugs are spliced into paths + env prefixes."""
    for args in (
        ["--character-name", "Assistant; rm -rf /", "--variant", "v"],
        ["--character-name", "Assistant", "--variant", "../evil"],
    ):
        proc = subprocess.run(
            ["bash", "scripts/issue1345_dispatch.sh", *args, "--dry-run"],
            capture_output=True,
            text=True,
            env=_clean_env({"REPO_ROOT": str(_REPO_ROOT)}),
            cwd=_REPO_ROOT,
            timeout=120,
        )
        assert proc.returncode != 0, args
        assert "must match" in proc.stderr, proc.stderr
