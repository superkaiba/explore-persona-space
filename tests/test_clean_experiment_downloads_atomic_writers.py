"""Direct success/failure-posture tests for the two batch-1 migrated writers
in ``scripts/clean_experiment_downloads.py`` (task #2336 plan v3 §4 step 4
batch-1 row; round-1 MF7 — grep showed NO existing test armed either writer,
so the ``atomic_replace`` migration would otherwise have landed unverified).

Per writer:

- SUCCESS: payload written and loadable, byte-identical to the pre-migration
  serialization (the §4 step 3 recipe keeps the serialization line VERBATIM,
  so the on-disk bytes must equal the old form's), zero ``*.tmp*`` residue —
  the glob covers BOTH the old ``<name>.tmp`` and the new process-unique
  ``<name>.<pid>.<uuid8>.tmp`` shapes (both end ``.tmp``).
- FAILURE posture (monkeypatched ``os.replace`` raising ``OSError``): the
  §4(k) fail-soft swallow contract is preserved — NO exception propagates,
  the destination is untouched, zero residue (``atomic_replace`` re-raises
  after unlinking its temp; the caller's ``except OSError`` swallows).
  ``_save_slurm_src_escalation_state`` additionally emits its stderr
  ``WARNING: writing slurm-src escalation dedup state failed`` line with the
  path + error class (the round-2 M1 never-silent contract).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import clean_experiment_downloads as ced  # noqa: E402


def _residue(root: Path) -> list[Path]:
    """Every leftover temp file under *root* (old AND new temp-name shapes)."""
    return sorted(p for p in root.rglob("*.tmp*") if p.is_file())


def _boom(*_a, **_k):  # signature-conformant stand-in for os.replace
    raise OSError("injected replace failure (#2336 batch-1 failure-posture test)")


# --------------------------------------------------------------------------
# _ScratchVerdictCache.save
# --------------------------------------------------------------------------


def _dirty_cache(tmp_path: Path) -> ced._ScratchVerdictCache:
    """A cache holding one real ``store()``-written entry (``_dirty=True``)."""
    cache = ced._ScratchVerdictCache(tmp_path / "cachedir" / "verdicts.json")
    cand = tmp_path / "cand-tree"
    cand.mkdir()
    stats = {"newest_mtime": 123.0, "total_bytes": 456}
    cache.store(cand, stats, "git-blob-proof", {"reason": "pass", "files": 3})
    assert cache._dirty is True
    return cache


def test_scratch_verdict_cache_save_success(tmp_path: Path) -> None:
    cache = _dirty_cache(tmp_path)
    cache.save()
    assert cache._dirty is False
    payload = json.loads(cache.path.read_text())
    assert list(payload.values()) == [
        {"evidence": "git-blob-proof", "detail": {"reason": "pass", "files": 3}}
    ]
    # Byte-identity with the pre-migration serialization (verbatim line).
    assert cache.path.read_bytes() == json.dumps(cache._data, sort_keys=True).encode()
    assert _residue(tmp_path) == []


def test_scratch_verdict_cache_save_failure_is_silent_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _dirty_cache(tmp_path)
    monkeypatch.setattr(os, "replace", _boom)
    cache.save()  # must NOT raise — the §4(k) silent swallow contract
    assert not cache.path.exists(), "destination must be untouched on failure"
    assert cache._dirty is True, "a failed save must not clear the dirty flag"
    assert _residue(tmp_path) == []


# --------------------------------------------------------------------------
# _save_slurm_src_escalation_state
# --------------------------------------------------------------------------


def test_save_slurm_src_escalation_state_success(tmp_path: Path) -> None:
    path = tmp_path / "state-dir" / "slurm-src-escalations.json"
    state = {"k1": {"ts": 1.0}, "k0": {"ts": 2.0}}
    ced._save_slurm_src_escalation_state(path, state)
    assert json.loads(path.read_text()) == state
    # Byte-identity with the pre-migration serialization (verbatim line).
    assert path.read_bytes() == json.dumps(state, sort_keys=True).encode()
    assert _residue(tmp_path) == []


def test_save_slurm_src_escalation_state_failure_warns_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "state-dir" / "slurm-src-escalations.json"
    monkeypatch.setattr(os, "replace", _boom)
    ced._save_slurm_src_escalation_state(path, {"k": 1})  # must NOT raise
    err = capsys.readouterr().err
    assert "WARNING: writing slurm-src escalation dedup state failed" in err
    assert str(path) in err
    assert "OSError" in err, "the failure line must carry the error class"
    assert not path.exists(), "destination must be untouched on failure"
    assert _residue(tmp_path) == []
