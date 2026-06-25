r"""End-to-end build smoke test for scripts/build_paper.py.

Builds a TEMPLATE-DERIVED paper (the real docs/papers/_template/issue_TEMPLATE.tex
filled with placeholders + the real preamble.tex copied alongside) through
build_paper.py --no-upload, then verifies it with verify_paper.py. This is the
test that catches the `\graphicspath`-before-`\input{preamble.tex}` ordering bug
(graphicx is loaded inside preamble.tex, so a mis-ordered template fails with
"Undefined control sequence \graphicspath").

Skipped off the build VM: requires pdflatex + bibtex + pandoc + a dashboard tree
with node_modules/.bin/tsx (the real sanitizer). When any is absent this is a
no-op, so the suite stays green on machines without the LaTeX toolchain.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO / "docs" / "papers" / "_template"


def _local_bin_on_path() -> dict[str, str]:
    """A PATH that includes ~/.local/bin (where the spike installed pandoc)."""
    env = dict(os.environ)
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    return env


def _which(name: str) -> str | None:
    return shutil.which(name, path=_local_bin_on_path()["PATH"])


def _dashboard_tsx_present() -> bool:
    """Mirror build_paper._resolve_dashboard_node_dir's lookup (worktree else root)."""
    candidates = [REPO / "dashboard"]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        ).stdout.strip()
        if common:
            candidates.append(Path(common).parent / "dashboard")
    except Exception:
        pass
    return any((d / "node_modules" / ".bin" / "tsx").exists() for d in candidates)


_TOOLS_PRESENT = (
    _which("pdflatex") is not None
    and _which("bibtex") is not None
    and _which("pandoc") is not None
    and _dashboard_tsx_present()
)

pytestmark = pytest.mark.skipif(
    not _TOOLS_PRESENT,
    reason="build toolchain absent (need pdflatex+bibtex+pandoc + dashboard tsx)",
)


def _tiny_png(path: Path) -> None:
    """Write a minimal valid 1x1 RGB PNG (so pandoc/pdflatex can embed it)."""

    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff")
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


# A real, citable .bib entry so the bibtex pass + bib-resolve check are exercised.
_BIB = """@misc{smoke2026,
  title = {A Smoke-Test Citation},
  author = {Test, Author},
  year = {2026}
}
"""

_FILL = {
    "TITLE": "A smoke-test claim about leakage",
    "ISSUE": "999999",
    "RUN_DATE": "2026-06-25",
    "MODEL": "Qwen2.5-7B",
    "ABSTRACT": (
        "We build a template-derived paper end to end to confirm the build "
        "pipeline compiles, renders, and verifies. This abstract is written out "
        "in full so the paper-stub and section checks have real prose to read."
    ),
    "INTRODUCTION": "This experiment exercises the production build path for EPS papers.",
    "METHODS": (
        "We fill the real template and preamble, compile multi-pass, render via "
        "pandoc + the Lua filter, and sanitize through the dashboard schema. "
        "We cite a placeholder reference \\cite{smoke2026}."
    ),
    "RESULTS": (
        "The build produces a PDF and a sanitized paper.html. "
        "\\includegraphics[width=0.4\\linewidth]{smoke_fig.png}"
    ),
    "DISCUSSION": "Nothing here binds interpretation; this is an infrastructure test.",
    "APPENDIX": "Full hyperparameters and worked examples would go here.",
    "GRAPHICSPATH": "./",
}


def _render_template() -> str:
    tex = (TEMPLATE_DIR / "issue_TEMPLATE.tex").read_text()
    for key, val in _FILL.items():
        tex = tex.replace("{{" + key + "}}", val)
    return tex


def test_template_derived_paper_builds_and_verifies(tmp_path: Path):
    jobname = f"issue_{_FILL['ISSUE']}"
    # Build under the real repo (build_paper.py + verify_paper.py resolve REPO,
    # the dashboard, and figure dirs relative to it); use a unique scratch dir
    # under docs/papers/ and remove it afterward.
    paper_dir = REPO / "docs" / "papers" / f"_smoketest_{os.getpid()}"
    if paper_dir.exists():
        shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True)
    try:
        (paper_dir / f"{jobname}.tex").write_text(_render_template())
        # The preamble is copied alongside (as a real authoring agent would) so
        # \input{preamble.tex} resolves in the paper dir.
        shutil.copy(TEMPLATE_DIR / "preamble.tex", paper_dir / "preamble.tex")
        (paper_dir / f"{jobname}.bib").write_text(_BIB)
        _tiny_png(paper_dir / "smoke_fig.png")

        env = _local_bin_on_path()
        rel = str(paper_dir.relative_to(REPO))

        build = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "build_paper.py"),
                "--paper-dir",
                rel,
                "--jobname",
                jobname,
                "--issue",
                _FILL["ISSUE"],
                "--no-upload",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            env=env,
        )
        assert build.returncode == 0, (
            "build_paper.py failed (proves the template ordering bug if it is a "
            f"\\graphicspath error):\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )
        pdf = paper_dir / f"{jobname}.pdf"
        assert pdf.exists() and pdf.stat().st_size > 0, "no PDF produced"
        assert (paper_dir / "paper.html").exists(), "no paper.html produced"
        assert (paper_dir / "paper_manifest.json").exists(), "no manifest produced"

        verify = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "verify_paper.py"),
                "--paper-dir",
                rel,
                "--jobname",
                jobname,
                "--issue",
                _FILL["ISSUE"],
                "--no-stub",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            env=env,
        )
        assert verify.returncode == 0, (
            f"verify_paper.py FAILed on the template-derived paper:\n"
            f"STDOUT:\n{verify.stdout}\nSTDERR:\n{verify.stderr}"
        )
    finally:
        shutil.rmtree(paper_dir, ignore_errors=True)
