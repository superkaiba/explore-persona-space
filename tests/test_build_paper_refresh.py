"""Unit tests for build_paper._refresh_template_owned_files.

Toolchain-free (no pdflatex/pandoc/tsx) so they run on every machine, unlike
the toolchain-gated end-to-end smoke test in test_build_paper_smoke.py. These
pin the refresh-on-every-build invariant (incident #657): a per-task
preamble.tex that predates a template change must be overwritten from the
canonical template at the start of each build, so a newly-template-shipped
macro/environment (e.g. epsexample) can never make the build fail against a
stale copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the builder as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_paper.py"
_spec = importlib.util.spec_from_file_location("build_paper", _SCRIPT)
build_paper = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["build_paper"] = build_paper
_spec.loader.exec_module(_spec and build_paper)  # type: ignore[union-attr]


def _template_dir(tmp_path: Path, preamble_text: str) -> Path:
    tdir = tmp_path / "_template"
    tdir.mkdir()
    (tdir / "preamble.tex").write_text(preamble_text)
    return tdir


def test_refresh_overwrites_stale_preamble(tmp_path, monkeypatch):
    """A drifted per-task preamble is overwritten with the template's content."""
    monkeypatch.setattr(build_paper, "TEMPLATE_DIR", _template_dir(tmp_path, "NEW\n"))
    pdir = tmp_path / "issue_X"
    pdir.mkdir()
    (pdir / "preamble.tex").write_text("STALE\n")

    build_paper._refresh_template_owned_files(pdir)

    assert (pdir / "preamble.tex").read_text() == "NEW\n"


def test_refresh_creates_missing_preamble(tmp_path, monkeypatch):
    """A paper dir without a preamble (the --paper-dir self-test path) gets one."""
    monkeypatch.setattr(build_paper, "TEMPLATE_DIR", _template_dir(tmp_path, "FROM_TEMPLATE\n"))
    pdir = tmp_path / "issue_X"
    pdir.mkdir()
    assert not (pdir / "preamble.tex").exists()

    build_paper._refresh_template_owned_files(pdir)

    assert (pdir / "preamble.tex").read_text() == "FROM_TEMPLATE\n"


def test_refresh_noop_when_already_current(tmp_path, monkeypatch):
    """An up-to-date per-task preamble survives byte-for-byte (idempotent)."""
    monkeypatch.setattr(build_paper, "TEMPLATE_DIR", _template_dir(tmp_path, "CURRENT\n"))
    pdir = tmp_path / "issue_X"
    pdir.mkdir()
    (pdir / "preamble.tex").write_text("CURRENT\n")

    build_paper._refresh_template_owned_files(pdir)

    assert (pdir / "preamble.tex").read_text() == "CURRENT\n"


def test_refresh_raises_when_template_missing(tmp_path, monkeypatch):
    """A missing template preamble is fail-loud, not a silent skip."""
    empty_template = tmp_path / "_template"
    empty_template.mkdir()
    monkeypatch.setattr(build_paper, "TEMPLATE_DIR", empty_template)
    pdir = tmp_path / "issue_X"
    pdir.mkdir()

    with pytest.raises(build_paper.BuildError, match="template preamble not found"):
        build_paper._refresh_template_owned_files(pdir)


def test_refresh_out_of_repo_dir_does_not_raise_valueerror(tmp_path, monkeypatch):
    """A drifted preamble in an out-of-REPO --paper-dir refreshes without crashing.

    The drift-notice prints `dst.relative_to(REPO)`, which raises ValueError when
    paper_dir is outside REPO; the helper must fall back to the absolute path.
    """
    monkeypatch.setattr(build_paper, "TEMPLATE_DIR", _template_dir(tmp_path, "NEW\n"))
    # An absolute path guaranteed outside the repo tree.
    pdir = tmp_path / "extern_paper"
    pdir.mkdir()
    (pdir / "preamble.tex").write_text("STALE\n")

    build_paper._refresh_template_owned_files(pdir)  # must not raise ValueError

    assert (pdir / "preamble.tex").read_text() == "NEW\n"
