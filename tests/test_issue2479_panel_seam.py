"""Byte-identity pins for the #2479 EPM_I2479_CHAR_PANEL_JSON panel seams.

Pins (issue #2479 U3):

(a) env ABSENT => ``issue1345_common``'s variant lists, the stager's
    ``CHAR_VARIANTS``, and the ladder-fill's ``CHAR_CHARACTERS`` /
    ``CHAR_VARIANTS`` all equal their hardcoded parent values EXACTLY.
(b) env SET to the committed ``eval_results/issue_2479/panel.json`` => the
    gen-side lists and the stager tuple EXTEND by exactly the panel's
    ``char_2479_``-prefixed variants (registry order), the ladder-fill
    swaps to the panel characters (the committed U2 semantics), and the
    inherited ``REGIME_SPECS`` r1/r2/r4/r4op entries are byte-identical to
    the env-absent state (registry variants never clobber them).
(c) the panel module's name-constraint validation fails LOUD on synthetic
    bad panels (substring-colliding name, FOIL name, non-single-token /
    non-capitalized names), and the shared env loader fails LOUD on a
    set-but-missing/malformed/schema-violating panel file.
(d) crash-fix round 2 (``unknown regime char_helios``): env SET => the
    ladder-fill's ``REGIME_SPECS``/``REGIME_LABEL`` carry lookup rows for
    BOTH namespaces — the legacy parent 16 ``char_*`` cells AND the
    ``char_2479_*`` panel cells — with the correct spec fields, while
    ``CHAR_VARIANTS`` (the --cells DEFAULT sweep feed) stays panel-only.

Hermetic: the env-state probes run in subprocesses (the seams read env at
IMPORT time) against in-repo files only — no network. The env-set fixture is
the COMMITTED panel.json; synthetic bad panels use tmp_path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
PANEL_JSON = REPO / "eval_results" / "issue_2479" / "panel.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2479_char_panel as panel_mod  # noqa: E402

# --- hardcoded parent values (byte-identity pins) ----------------------------

HARDCODED_PAIRED = (
    "conversation_paired_stories",
    "conversation_paired_stories_assistant",
    "conversation_paired_stories_assistant_base",
    "char_helios",
    "char_helios_base",
    "char_wren",
    "char_wren_base",
    "char_dana",
    "char_dana_base",
    "char_vex",
    "char_vex_base",
)
HARDCODED_ONPOLICY = (
    "onpolicy_assistant_story",
    "char_helios_op",
    "char_helios_op_base",
    "char_wren_op",
    "char_wren_op_base",
    "char_dana_op",
    "char_dana_op_base",
    "char_vex_op",
    "char_vex_op_base",
)
HARDCODED_BASE_PAIRED = (
    "conversation_paired_stories_assistant_base",
    "char_helios_base",
    "char_wren_base",
    "char_dana_base",
    "char_vex_base",
    "char_helios_op_base",
    "char_wren_op_base",
    "char_dana_op_base",
    "char_vex_op_base",
)
HARDCODED_CHAR_VARIANTS = tuple(
    f"char_{ch}{suf}"
    for ch in ("helios", "wren", "dana", "vex")
    for suf in ("", "_op", "_base", "_op_base")
)
HARDCODED_CHAR_CHARACTERS = ("helios", "wren", "dana", "vex")

_PROBE = """
import json, sys
sys.path.insert(0, {scripts!r})
import issue1345_common as c
import issue1345_stage_char_stories as st
import issue1345_story_char_ladder_fill as lf
base_specs = {{k: lf.REGIME_SPECS[k] for k in ("r1", "r2", "r4", "r4op")}}
# Legacy namespace constructed LOCALLY (not lf.LEGACY_CHAR_VARIANTS) so the
# probe itself also runs against pre-fix module code — the fails-pre-fix
# property of the (d) pins lives in the assertions, not a probe AttributeError.
legacy = tuple(
    "char_" + ch + suf
    for ch in ("helios", "wren", "dana", "vex")
    for suf in ("", "_op", "_base", "_op_base")
)
print("PANEL_SEAM_JSON::" + json.dumps({{
    "paired": list(c.PAIRED_STORIES_VARIANTS),
    "onpolicy": list(c.ONPOLICY_STORY_VARIANTS),
    "base_paired": list(c.BASE_PAIRED_STORIES_VARIANTS),
    "stager_char_variants": list(st.CHAR_VARIANTS),
    "ladder_char_characters": list(lf.CHAR_CHARACTERS),
    "ladder_char_variants": list(lf.CHAR_VARIANTS),
    "ladder_base_regime_specs": base_specs,
    "ladder_legacy_specs": {{v: lf.REGIME_SPECS.get(v) for v in legacy}},
    "ladder_panel_ns_specs": {{v: lf.REGIME_SPECS.get(v) for v in lf.CHAR_VARIANTS}},
    "ladder_legacy_labels": [v for v in legacy if lf.REGIME_LABEL.get(v) == v],
    "ladder_default_cells_instruct": lf._default_cells("instruct"),
}}))
"""


def _probe(env_value: str | None) -> dict:
    """Import the three seam-bearing modules in a subprocess; return the dict."""
    env = os.environ.copy()
    for k in (
        "EPM_I2479_CHAR_PANEL_JSON",
        "EPM_I1345_VARIANT",
        "EPM_STORY_CHARACTER_NAME",
        "EPM_I1345_PERSONA_DESC",
    ):
        env.pop(k, None)
    if env_value is not None:
        env["EPM_I2479_CHAR_PANEL_JSON"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(scripts=str(SCRIPTS))],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=600,
    )
    assert out.returncode == 0, f"probe import failed:\n{out.stderr[-4000:]}"
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("PANEL_SEAM_JSON::")]
    assert len(lines) == 1, out.stdout[-2000:]
    return json.loads(lines[0].removeprefix("PANEL_SEAM_JSON::"))


@pytest.fixture(scope="module")
def env_absent() -> dict:
    return _probe(None)


@pytest.fixture(scope="module")
def env_set() -> dict:
    assert PANEL_JSON.is_file(), f"committed panel missing: {PANEL_JSON}"
    return _probe(str(PANEL_JSON))


@pytest.fixture(scope="module")
def panel_rows() -> list[dict]:
    return json.loads(PANEL_JSON.read_text())


# --- (a) env ABSENT: byte-identical hardcoded values --------------------------


def test_env_absent_byte_identity(env_absent):
    assert tuple(env_absent["paired"]) == HARDCODED_PAIRED
    assert tuple(env_absent["onpolicy"]) == HARDCODED_ONPOLICY
    assert tuple(env_absent["base_paired"]) == HARDCODED_BASE_PAIRED
    assert tuple(env_absent["stager_char_variants"]) == HARDCODED_CHAR_VARIANTS
    assert tuple(env_absent["ladder_char_characters"]) == HARDCODED_CHAR_CHARACTERS
    assert tuple(env_absent["ladder_char_variants"]) == HARDCODED_CHAR_VARIANTS


# --- (b) env SET: extend by exactly the panel variants ------------------------


def test_env_set_extends_gen_side_lists(env_set, panel_rows):
    inserted = tuple(r["variant_inserted"] for r in panel_rows if r["variant_inserted"])
    ops = tuple(r["variant_op"] for r in panel_rows)
    assert tuple(env_set["paired"]) == HARDCODED_PAIRED + inserted
    assert tuple(env_set["onpolicy"]) == HARDCODED_ONPOLICY + ops
    # instruct-only panel: the base-measured list is NEVER extended.
    assert tuple(env_set["base_paired"]) == HARDCODED_BASE_PAIRED


def test_env_set_extends_stager(env_set, panel_rows):
    ext = tuple(v for r in panel_rows for v in (r["variant_op"], r["variant_inserted"]) if v)
    assert tuple(env_set["stager_char_variants"]) == HARDCODED_CHAR_VARIANTS + ext


def test_env_set_ladder_swaps_to_panel(env_set, panel_rows):
    # U2 committed semantics: under the registry the ladder-fill REPLACES the
    # parent 4-character panel with the registry characters/variants.
    assert tuple(env_set["ladder_char_characters"]) == tuple(r["name"] for r in panel_rows)
    assert tuple(env_set["ladder_char_variants"]) == tuple(
        v for r in panel_rows for v in (r["variant_op"], r["variant_inserted"]) if v
    )


def test_env_set_prefix_and_regime_specs(env_set, env_absent, panel_rows):
    # Every extension variant is char_2479_-prefixed (structural guard against
    # REGIME_SPECS clobbering), and the inherited r1/r2/r4/r4op spec entries
    # are byte-identical across env states.
    for r in panel_rows:
        assert r["variant_op"].startswith("char_2479_")
        if r["variant_inserted"]:
            assert r["variant_inserted"].startswith("char_2479_")
    assert env_set["ladder_base_regime_specs"] == env_absent["ladder_base_regime_specs"]
    base_keys = {"r1", "r2", "r4", "r4op"}
    all_panel_variants = {
        v for r in panel_rows for v in (r["variant_op"], r["variant_inserted"]) if v
    }
    assert not (all_panel_variants & base_keys)


# --- (d) BOTH regime-spec namespaces resolve under the panel env --------------
# Crash-fix round 2: relaunch P0 died at leg toyfit-newcell-store with
# `AssertionError: unknown regime char_helios` — the panel branch registered
# ONLY the char_2479_* rows while the wrapper's P0 legs 5-8 pass PARENT names.


def _expected_spec(v: str) -> dict:
    """The _char_specs() row shape for one character cell (mirrors the module)."""
    return {
        "format_key": "stories_paired_op" if "_op" in v else "stories_paired",
        "subdir": f"{v}_turnstore",
        "turn": 0,
        "model": "pretrained" if v.endswith("_base") else "instruct",
        "cache_key": v,
    }


def test_env_set_legacy_regime_rows_present(env_set):
    # FAILS PRE-FIX: with the panel env set the old code path lacked the
    # legacy rows entirely (REGIME_SPECS.get("char_helios") was None).
    for v in HARDCODED_CHAR_VARIANTS:
        assert env_set["ladder_legacy_specs"][v] == _expected_spec(v), v
    assert set(env_set["ladder_legacy_labels"]) == set(HARDCODED_CHAR_VARIANTS)


def test_env_set_panel_regime_rows_present(env_set, panel_rows):
    variants = [v for r in panel_rows for v in (r["variant_op"], r["variant_inserted"]) if v]
    for v in variants:
        assert env_set["ladder_panel_ns_specs"][v] == _expected_spec(v), v


def test_env_set_char_variants_panel_only(env_set):
    # (b) CHAR_VARIANTS under the panel env contains ONLY panel variants —
    # the legacy rows are lookup-only, never sweep members.
    assert not set(env_set["ladder_char_variants"]) & set(HARDCODED_CHAR_VARIANTS)
    assert all(v.startswith("char_2479_") for v in env_set["ladder_char_variants"])


def test_env_set_default_cells_sweep_panel_only(env_set, panel_rows):
    # (c) the effective --cells DEFAULT sweep (main() routes through the
    # production resolver lf._default_cells — probed directly, r10 addendum)
    # is unchanged panel-only, in registry order.
    expected = [v for r in panel_rows for v in (r["variant_op"], r["variant_inserted"]) if v]
    assert env_set["ladder_default_cells_instruct"] == expected


def test_env_set_legacy_collision_rejected(tmp_path):
    # r10 reconciler addendum: a panel row naming a legacy parent variant id
    # must fail loud at IMPORT, before registry construction — the loader's
    # prefix/suffix conventions alone accept it (the r10 Codex finding).
    committed = json.loads(PANEL_JSON.read_text())
    row = dict(committed[0])
    row["variant_op"] = "char_helios_op"
    bad = tmp_path / "panel_bad.json"
    bad.write_text(json.dumps([row]))
    env = os.environ.copy()
    for k in (
        "EPM_I2479_CHAR_PANEL_JSON",
        "EPM_I1345_VARIANT",
        "EPM_STORY_CHARACTER_NAME",
        "EPM_I1345_PERSONA_DESC",
    ):
        env.pop(k, None)
    env["EPM_I2479_CHAR_PANEL_JSON"] = str(bad)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(scripts=str(SCRIPTS))],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=600,
    )
    assert out.returncode != 0, "legacy-colliding panel id must refuse at import"
    assert "collide with legacy parent cells" in out.stderr


def test_env_absent_registration_idempotent(env_absent):
    # env-unset: CHAR_VARIANTS == LEGACY_CHAR_VARIANTS, so the unconditional
    # double registration is an idempotent no-op with the same spec rows.
    for v in HARDCODED_CHAR_VARIANTS:
        assert env_absent["ladder_legacy_specs"][v] == _expected_spec(v), v
    assert set(env_absent["ladder_legacy_labels"]) == set(HARDCODED_CHAR_VARIANTS)


# --- committed panel <-> module registry drift pin ----------------------------


def test_committed_panel_matches_module_registry(panel_rows):
    assert panel_rows == [dict(r) for r in panel_mod.PANEL]
    assert len(panel_rows) == 16
    assert sum(1 for r in panel_rows if r["variant_inserted"]) == 8


# --- (c) name-constraint + loader fail-loud -----------------------------------


def test_validate_rejects_substring_collision():
    with pytest.raises(ValueError, match="substring"):
        panel_mod.validate_display_names(["Gus", "Gustav"])


def test_validate_rejects_foil_names():
    with pytest.raises(ValueError, match="FOIL"):
        panel_mod.validate_display_names(["Sam"])
    with pytest.raises(ValueError, match="FOIL"):
        panel_mod.validate_display_names(["Mara"])


def test_validate_rejects_malformed_tokens():
    with pytest.raises(ValueError, match="single alphabetic capitalized"):
        panel_mod.validate_display_names(["Big Gus"])
    with pytest.raises(ValueError, match="single alphabetic capitalized"):
        panel_mod.validate_display_names(["iris"])


def test_loader_fail_loud(tmp_path, monkeypatch):
    env = panel_mod.CHAR_PANEL_ENV
    # unset/empty -> None (the byte-identity fail-safe)
    monkeypatch.delenv(env, raising=False)
    assert panel_mod.load_char_panel_env() is None
    # set but missing -> raises
    monkeypatch.setenv(env, str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError):
        panel_mod.load_char_panel_env()
    # malformed JSON -> raises
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv(env, str(bad))
    with pytest.raises(ValueError, match="malformed"):
        panel_mod.load_char_panel_env()
    # schema violation: variant id without the char_ prefix -> raises
    rows = [
        {
            "name": "iris",
            "variant_op": "r4op",
            "variant_inserted": None,
            "design_band": "A",
        }
    ]
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps(rows))
    monkeypatch.setenv(env, str(bad2))
    with pytest.raises(ValueError, match="char_"):
        panel_mod.load_char_panel_env()
