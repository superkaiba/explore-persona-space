"""Full-vs-smoke built-data guard in run_issue543_ratio (#543 round-2 standing rec).

``_check_data_built()`` must NOT early-return "already built" on smoke-build
leftovers (a ``--smoke`` mix build writes the SAME paths with tiny row
counts): it requires the manifest to parse as a full build (``smoke=False``,
``total_rows_per_arm == TOTAL_ROWS``) plus per-file line-count spot checks
(arm train.jsonl == TOTAL_ROWS rows, probe files == N_PROBE_ROWS rows).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_issue543_ratio as mod  # noqa: E402


def _write_layout(
    tmp_path: Path, *, smoke: bool, total_rows: int, n_train_lines: int, n_probe_lines: int
) -> tuple[Path, Path, Path]:
    """Fabricate a MIXES_DIR/PROBES_DIR layout with trivial JSONL rows."""
    mixes = tmp_path / "mixes"
    probes = tmp_path / "probes"
    manifest = mixes / "manifest.json"
    probes.mkdir()
    row = json.dumps({"text": "x"}) + "\n"
    for arm in mod.ARMS:
        arm_dir = mixes / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "train.jsonl").write_text(row * n_train_lines)
    for fname in mod.PROBE_LOG_PREFIXES:
        (probes / fname).write_text(row * n_probe_lines)
    manifest.write_text(json.dumps({"smoke": smoke, "total_rows_per_arm": total_rows}))
    return mixes, probes, manifest


@pytest.fixture
def patched_layout(tmp_path, monkeypatch):
    """Write a fixture layout and point the module's path constants at it."""

    def apply(**kwargs) -> None:
        mixes, probes, manifest = _write_layout(tmp_path, **kwargs)
        monkeypatch.setattr(mod, "MIXES_DIR", mixes)
        monkeypatch.setattr(mod, "PROBES_DIR", probes)
        monkeypatch.setattr(mod, "MIX_MANIFEST_PATH", manifest)

    return apply


def test_smoke_artifacts_read_as_not_built(patched_layout):
    """Smoke-shaped manifest + 48-row train files must fall through to rebuild."""
    patched_layout(smoke=True, total_rows=48, n_train_lines=48, n_probe_lines=3)
    assert mod._check_data_built() is False


def test_full_shaped_fixture_reads_as_built(patched_layout):
    """A full-shaped fixture (smoke=False, TOTAL_ROWS rows, N_PROBE_ROWS probes) passes."""
    patched_layout(
        smoke=False,
        total_rows=mod.TOTAL_ROWS,
        n_train_lines=mod.TOTAL_ROWS,
        n_probe_lines=mod.N_PROBE_ROWS,
    )
    assert mod._check_data_built() is True
