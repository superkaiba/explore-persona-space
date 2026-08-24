"""#2479 r2 regression pins for the gen script's manifest binding + CLI targets.

Pins two round-1 codex blockers on ``scripts/issue1345_gen_stories_paired.py``:

(a) ``op-powered-cli-ignored`` — ``--op-powered`` previously pinned the parent
    constants (2200/2000) regardless of the wrapper's explicit
    ``--n-stories 1600 --yield-floor 800``; ``resolve_gen_targets`` now binds
    explicit CLI values in BOTH modes with mode constants as defaults.
(b) ``manifest-and-reservation-disconnected`` (gen half) — production gen never
    read the committed ``panel_manifest.json``; ``restrict_pool_to_manifest``
    now binds panel-cell pools to the registered ``sample_conv_ids`` (fail-loud
    on a registered id missing from the pool) and folds the manifest sha into
    the bundle fingerprint via ``apply_manifest_fp_suffix``. Non-panel variants
    stay byte-identical to the parent.

Hermetic: tmp-path manifests + monkeypatched ``c.VARIANT``; the one committed
artifact read is ``eval_results/issue_2479/panel_manifest.json`` (conv-id
digests only — no corpus text). No network, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
MANIFEST_JSON = REPO / "eval_results" / "issue_2479" / "panel_manifest.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_gen_stories_paired as gp  # noqa: E402

PANEL_VARIANT = "char_2479_testchar"


def _write_manifest(path: Path, ids: list[str]) -> Path:
    path.write_text(json.dumps({"sample_conv_ids": ids, "n_sample": len(ids)}))
    return path


# --- (a) per-mode CLI target resolution --------------------------------------


def test_op_powered_explicit_cli_binds():
    """Pre-fix regression: --op-powered ignored --n-stories/--yield-floor."""
    assert gp.resolve_gen_targets(True, 1600, 800) == (1600, 800)


def test_paired_explicit_cli_binds():
    assert gp.resolve_gen_targets(False, 123, 45) == (123, 45)


def test_defaults_are_mode_constants():
    c = gp.c
    assert gp.resolve_gen_targets(True, None, None) == (
        c.N_ONPOLICY_STORY_TARGET,
        c.ONPOLICY_STORY_YIELD_FLOOR,
    )
    assert gp.resolve_gen_targets(False, None, None) == (
        c.N_STORIES_PAIRED_TARGET,
        c.STORY_PAIRED_YIELD_FLOOR,
    )


# --- (b) manifest pool binding ------------------------------------------------


def test_non_panel_variant_is_byte_identical_passthrough():
    pool = [{"conv_id": "a"}, {"conv_id": "b"}]
    out, meta = gp.restrict_pool_to_manifest(pool)
    assert out is pool and meta is None
    assert gp.apply_manifest_fp_suffix("fp123", None) == "fp123"


def test_panel_variant_restricts_to_registered_ids(monkeypatch, tmp_path):
    mp = _write_manifest(tmp_path / "m.json", ["a", "b", "c"])
    monkeypatch.setattr(gp.c, "VARIANT", PANEL_VARIANT)
    monkeypatch.setenv(gp.I2479_MANIFEST_ENV, str(mp))
    pool = [{"conv_id": k} for k in ("a", "b", "c", "d", "e")]
    out, meta = gp.restrict_pool_to_manifest(pool)
    assert [r["conv_id"] for r in out] == ["a", "b", "c"]  # order preserved
    assert meta is not None
    assert meta["n_manifest_registered"] == 3
    assert meta["n_pool_before_manifest_restrict"] == 5
    assert meta["n_pool_after_manifest_restrict"] == 3
    sha = meta["panel_manifest_sha256"]
    assert len(sha) == 64
    assert gp.apply_manifest_fp_suffix("fp123", meta) == f"fp123-m{sha[:12]}"


def test_panel_variant_fails_loud_on_missing_registered_id(monkeypatch, tmp_path):
    mp = _write_manifest(tmp_path / "m.json", ["a", "b", "zz_not_in_pool"])
    monkeypatch.setattr(gp.c, "VARIANT", PANEL_VARIANT)
    monkeypatch.setenv(gp.I2479_MANIFEST_ENV, str(mp))
    pool = [{"conv_id": "a"}, {"conv_id": "b"}]
    with pytest.raises(AssertionError, match="zz_not_in_pool"):
        gp.restrict_pool_to_manifest(pool)


def test_panel_variant_fails_loud_on_absent_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(gp.c, "VARIANT", PANEL_VARIANT)
    monkeypatch.setenv(gp.I2479_MANIFEST_ENV, str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="panel manifest"):
        gp.restrict_pool_to_manifest([{"conv_id": "a"}])


def test_panel_variant_fails_loud_on_duplicate_ids(monkeypatch, tmp_path):
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps({"sample_conv_ids": ["a", "a", "b"], "n_sample": 3}))
    monkeypatch.setattr(gp.c, "VARIANT", PANEL_VARIANT)
    monkeypatch.setenv(gp.I2479_MANIFEST_ENV, str(mp))
    with pytest.raises(AssertionError, match="duplicate"):
        gp.restrict_pool_to_manifest([{"conv_id": "a"}, {"conv_id": "b"}])


def test_committed_manifest_schema():
    """The default (env-absent) manifest path resolves + carries 1600 unique ids."""
    m = json.loads(MANIFEST_JSON.read_text())
    ids = m["sample_conv_ids"]
    assert m["n_sample"] == 1600 and len(ids) == 1600
    assert len({str(x) for x in ids}) == 1600
    reserved = set(map(str, m["axis_reservation_conv_ids"]))
    assert len(reserved) == m["n_reservation"] == 250
    # axis reservation ⊂ sample (the fill-side exclusion in load_regime_xy
    # subtracts these from panel-cell rows; disjointness there requires
    # membership here).
    assert reserved <= {str(x) for x in ids}


def test_op_companion_refused_on_panel_variant(monkeypatch):
    """--op-companion is parent-only; a panel cell must fail loud pre-tokenizer."""
    monkeypatch.setattr(gp.c, "VARIANT", PANEL_VARIANT)
    monkeypatch.setattr(sys, "argv", ["issue1345_gen_stories_paired.py", "--op-companion"])
    with pytest.raises(AssertionError, match="parent-only control mode"):
        gp.main()
