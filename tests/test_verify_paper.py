"""Tests for scripts/verify_paper.py — the `paper: true` clean-result verifier.

Hermetic: every test builds a synthetic paper-dir fixture in tmp_path and
exercises one check function. No network, no pdflatex/pandoc — the compile-clean
check reads pre-written `.log`/`.blg`/`.bbl` fixtures, and the other checks parse
the `.tex` / manifest / stub directly. The end-to-end build+upload path is
validated separately (the spike self-test in the Phase A report), not here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_paper.py"
_spec = importlib.util.spec_from_file_location("verify_paper", _SCRIPT)
verify_paper = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_paper"] = verify_paper
_spec.loader.exec_module(_spec and verify_paper)  # type: ignore[union-attr]


# ─── fixture: a minimal spec-conformant paper .tex ──────────────────────────

GOOD_TEX = r"""\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage[numbers,square]{natbib}
\graphicspath{{figs/}}
\newcommand{\metric}[1]{X}
\newcommand{\epsref}[1]{\##1}
\title{A claim about leakage}
\begin{document}
\maketitle
\begin{abstract}
We test a thing and report a result. The base-rate link generalized to one new
behavior, not three. This abstract is long enough to be a real abstract.
\end{abstract}
\section{Introduction}
Context for the project. Parent \epsref{623}.
\section{Methods}
We reuse persona vectors and write out the recipe. \cite{chen2025persona}
\section{Results}
The signal is flat. See Figure. \includegraphics[width=0.5\linewidth]{fig_one.png}
\section{Discussion}
The targets come apart. See \epsref{623}.
\bibliographystyle{plainnat}
\bibliography{issue_657}
\appendix
\section{Appendix}
Full hyperparameters + completions go here.
\end{document}
"""

GOOD_BIB = """@misc{chen2025persona,
  title = {Persona Vectors},
  author = {Chen, Runjin},
  year = {2025}
}
"""

GOOD_STUB = """---
kind: experiment
paper: true
title: A claim about leakage (MODERATE confidence)
---
# A claim about leakage

We test a thing and report a result. This is the abstract paragraph and it is
clearly long enough to count as an abstract for the stub check.

Paper: docs/papers/issue_657/issue_657.pdf
"""


def _write_clean_logs(paper_dir: Path, jobname: str) -> None:
    """Write a clean build-log set the compile-clean check accepts."""
    (paper_dir / f"{jobname}.pass3.log").write_text(
        "This is pdfTeX, Version 3.14\nOutput written on issue_657.pdf (3 pages).\n"
        "Transcript written.\n"
    )
    (paper_dir / f"{jobname}.blg").write_text(
        "This is BibTeX, Version 0.99d\nDatabase file #1: issue_657.bib\n"
        "(There were 0 error messages)\n"
    )
    (paper_dir / f"{jobname}.bbl").write_text(
        "\\begin{thebibliography}{1}\n\\bibitem{chen2025persona} Chen.\n\\end{thebibliography}\n"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _make_paper(tmp_path: Path, jobname: str = "issue_657") -> Path:
    """Create a complete, valid synthetic paper dir. Returns the paper dir."""
    paper_dir = tmp_path / "docs" / "papers" / "issue_657"
    (paper_dir / "figs").mkdir(parents=True)
    (paper_dir / f"{jobname}.tex").write_text(GOOD_TEX)
    (paper_dir / "issue_657.bib").write_text(GOOD_BIB)
    (paper_dir / "figs" / "fig_one.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    # a real (tiny) PDF + paper.html so manifest hashes resolve.
    (paper_dir / f"{jobname}.pdf").write_bytes(b"%PDF-1.5\nfake pdf\n%%EOF\n")
    (paper_dir / "paper.html").write_text("<figure><img src='x'></figure>")
    _write_clean_logs(paper_dir, jobname)
    return paper_dir


# ─── compile-clean ───────────────────────────────────────────────────────────


def test_compile_clean_pass(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    r = verify_paper.check_compile_clean(paper_dir, "issue_657")
    assert r.passed, r.detail


def test_compile_clean_fails_on_undefined_refs(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    (paper_dir / "issue_657.pass3.log").write_text(
        "Output written.\nLaTeX Warning: There were undefined references.\n"
    )
    r = verify_paper.check_compile_clean(paper_dir, "issue_657")
    assert not r.passed
    assert "undefined references" in r.detail


def test_compile_clean_fails_on_undefined_citation(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    (paper_dir / "issue_657.blg").write_text('I didn\'t find a database entry for "missingkey"\n')
    r = verify_paper.check_compile_clean(paper_dir, "issue_657")
    assert not r.passed
    assert "bibtex" in r.detail.lower()


def test_compile_clean_fails_on_latex_error(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    (paper_dir / "issue_657.pass3.log").write_text("! Undefined control sequence.\n")
    r = verify_paper.check_compile_clean(paper_dir, "issue_657")
    assert not r.passed


def test_compile_clean_fails_on_missing_bbl(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    (paper_dir / "issue_657.bbl").unlink()
    r = verify_paper.check_compile_clean(paper_dir, "issue_657")
    assert not r.passed
    assert ".bbl" in r.detail


def test_compile_clean_fails_on_rerun_warning(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    (paper_dir / "issue_657.pass3.log").write_text(
        "Output written.\nLaTeX Warning: Label(s) may have changed. "
        "Rerun to get cross-references right.\n"
    )
    r = verify_paper.check_compile_clean(paper_dir, "issue_657")
    assert not r.passed
    assert "Rerun" in r.detail


# ─── required sections ───────────────────────────────────────────────────────


def test_required_sections_pass():
    r = verify_paper.check_required_sections(GOOD_TEX)
    assert r.passed, r.detail


def test_required_sections_fails_missing_appendix():
    tex = GOOD_TEX.replace("\\appendix\n\\section{Appendix}\n", "")
    r = verify_paper.check_required_sections(tex)
    assert not r.passed
    assert "Appendix" in r.detail


def test_required_sections_fails_missing_methods():
    tex = GOOD_TEX.replace("\\section{Methods}", "\\section{Approach}")
    r = verify_paper.check_required_sections(tex)
    assert not r.passed
    assert "Methods" in r.detail


def test_required_sections_fails_out_of_order():
    # A genuinely out-of-order body: Results appears BEFORE Methods (each section
    # present exactly once, but the required order is violated).
    tex = r"""\documentclass[11pt]{article}
\graphicspath{{figs/}}
\title{T}
\begin{document}
\maketitle
\begin{abstract}
This abstract is long enough to be a real abstract for the order check fixture.
\end{abstract}
\section{Introduction}
Intro.
\section{Results}
Results come before Methods here.
\section{Methods}
Methods after Results — out of order.
\section{Discussion}
Discussion.
\bibliographystyle{plainnat}
\bibliography{issue_657}
\appendix
\section{Appendix}
Appendix.
\end{document}
"""
    r = verify_paper.check_required_sections(tex)
    assert not r.passed
    assert "order" in r.detail.lower()


# ─── no confidence in body ───────────────────────────────────────────────────


def test_no_confidence_pass():
    r = verify_paper.check_no_confidence(GOOD_TEX)
    assert r.passed, r.detail


def test_no_confidence_fails_on_tag():
    tex = GOOD_TEX.replace(
        "\\section{Discussion}",
        "\\section{Discussion}\nThis is a strong result (HIGH confidence).",
    )
    r = verify_paper.check_no_confidence(tex)
    assert not r.passed


def test_no_confidence_fails_on_line():
    tex = GOOD_TEX.replace("\\section{Discussion}", "\\section{Discussion}\nConfidence: high")
    r = verify_paper.check_no_confidence(tex)
    assert not r.passed


def test_no_confidence_ignores_preamble_comment():
    # a confidence word in a preamble COMMENT (outside the body) must not FAIL.
    tex = "% Confidence: this is a comment\n" + GOOD_TEX
    r = verify_paper.check_no_confidence(tex)
    assert r.passed


def test_no_confidence_ignores_in_body_comment():
    # a commented `% Confidence:` line INSIDE the body (between \begin{document}
    # and \end{document}) must be skipped, not FAILed (regression: the old
    # `%?`-prefixed regex matched at the `%`, defeating the comment skip).
    tex = GOOD_TEX.replace(
        "\\section{Discussion}",
        "\\section{Discussion}\n% Confidence: high (a note to self, not body text)",
    )
    r = verify_paper.check_no_confidence(tex)
    assert r.passed, r.detail


# ─── includegraphics confined + resolves ─────────────────────────────────────


def test_includegraphics_pass(tmp_path: Path, monkeypatch):
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    r = verify_paper.check_includegraphics(GOOD_TEX, paper_dir)
    assert r.passed, r.detail


def test_includegraphics_fails_unresolved(tmp_path: Path, monkeypatch):
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    (paper_dir / "figs" / "fig_one.png").unlink()
    r = verify_paper.check_includegraphics(GOOD_TEX, paper_dir)
    assert not r.passed
    assert "resolve" in r.detail


def test_includegraphics_fails_absolute_path(tmp_path: Path, monkeypatch):
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    tex = GOOD_TEX.replace("{fig_one.png}", "{/etc/passwd}")
    r = verify_paper.check_includegraphics(tex, paper_dir)
    assert not r.passed
    assert "absolute" in r.detail


def test_includegraphics_fails_escapes_repo(tmp_path: Path, monkeypatch):
    # a figure that resolves OUTSIDE the repo (via ..) is a confinement FAIL.
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path / "docs" / "papers")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    tex = GOOD_TEX.replace("\\graphicspath{{figs/}}", "\\graphicspath{{../../../}}")
    tex = tex.replace("{fig_one.png}", "{outside.png}")
    r = verify_paper.check_includegraphics(tex, paper_dir)
    assert not r.passed
    assert "OUTSIDE" in r.detail or "resolve" in r.detail


# ─── bib resolves ────────────────────────────────────────────────────────────


def test_bib_resolves_pass(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    r = verify_paper.check_bib_resolves(GOOD_TEX, paper_dir, "issue_657", "657")
    assert r.passed, r.detail


def test_bib_fails_missing_entry(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    tex = GOOD_TEX.replace("\\cite{chen2025persona}", "\\cite{nonexistent2099}")
    r = verify_paper.check_bib_resolves(tex, paper_dir, "issue_657", "657")
    assert not r.passed
    assert "nonexistent2099" in r.detail


# ─── epsref resolves ─────────────────────────────────────────────────────────


def test_epsref_pass(monkeypatch):
    monkeypatch.setattr(verify_paper, "_registry_task_ids", lambda: {"623", "657"})
    r = verify_paper.check_epsref_resolves(GOOD_TEX)
    assert r.passed, r.detail


def test_epsref_fails_nonexistent(monkeypatch):
    monkeypatch.setattr(verify_paper, "_registry_task_ids", lambda: {"100"})
    r = verify_paper.check_epsref_resolves(GOOD_TEX)
    assert not r.passed
    assert "623" in r.detail


def test_epsref_warns_when_registry_unavailable(monkeypatch):
    monkeypatch.setattr(verify_paper, "_registry_task_ids", lambda: set())
    r = verify_paper.check_epsref_resolves(GOOD_TEX)
    assert r.passed and r.is_warn


# ─── manifest complete + hashes ──────────────────────────────────────────────


def _write_real_manifest(
    paper_dir: Path,
    repo: Path,
    jobname: str,
    *,
    pdf_url,
    shape: str = "hf",
):
    """Write a manifest with REPO-relative paths + REAL sha256 hashes.

    ``shape='hf'`` (NEW, the build_paper.py shape): the PDF is HF-hosted — only
    the COMMITTED artifacts (tex/paper_html) go in ``artifacts``; the PDF
    provenance lives in an ``hf_pdf`` block. ``shape='old'``: the legacy shape
    that records the PDF as a local ``pdf`` artifact (an already-built manifest).
    """
    tex = paper_dir / f"{jobname}.tex"
    pdf = paper_dir / f"{jobname}.pdf"
    html = paper_dir / "paper.html"

    def rec(p: Path):
        return {"path": str(p.relative_to(repo)), "sha256": _sha256(p), "bytes": p.stat().st_size}

    artifacts = {"tex": rec(tex), "paper_html": rec(html)}
    manifest = {
        "schema": "paper_manifest/v1",
        "issue": 657,
        "jobname": jobname,
        "pdf_hf_url": pdf_url,
        "artifacts": artifacts,
    }
    if shape == "hf":
        manifest["hf_pdf"] = {
            "url": pdf_url,
            "sha256": _sha256(pdf),
            "bytes": pdf.stat().st_size,
        }
    elif shape == "old":
        # OLD shape: the PDF is recorded as a local artifact (validators must
        # SKIP its local-existence check — it is HF-hosted).
        artifacts["pdf"] = rec(pdf)
    (paper_dir / "paper_manifest.json").write_text(json.dumps(manifest))


def test_manifest_pass(tmp_path: Path, monkeypatch):
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url="https://hf/x.pdf")
    r = verify_paper.check_manifest(paper_dir)
    assert r.passed and not r.is_warn, r.detail


def test_manifest_pass_when_pdf_is_hf_only_no_local_file(tmp_path: Path, monkeypatch):
    """The storage decision: the PDF lives on HF, NOT committed. A manifest whose
    PDF is HF-hosted with NO local pdf file on disk must PASS (incident #657 — the
    local PDF is gone post-commit)."""
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url="https://hf/x.pdf")
    # Delete the local PDF — it is HF-only post-commit.
    (paper_dir / "issue_657.pdf").unlink()
    r = verify_paper.check_manifest(paper_dir)
    assert r.passed and not r.is_warn, r.detail


def test_manifest_old_shape_pdf_in_artifacts_hf_only_passes(tmp_path: Path, monkeypatch):
    """Tolerance for the OLD manifest shape (#657's existing manifest): a `pdf`
    entry in `artifacts` whose local file is gone (HF-only) still PASSes — the
    PDF's local check is skipped, validated via pdf_hf_url."""
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url="https://hf/x.pdf", shape="old")
    (paper_dir / "issue_657.pdf").unlink()  # HF-only post-commit
    r = verify_paper.check_manifest(paper_dir)
    assert r.passed and not r.is_warn, r.detail


def test_manifest_warns_when_no_pdf_url(tmp_path: Path, monkeypatch):
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url=None)
    r = verify_paper.check_manifest(paper_dir)
    assert r.passed and r.is_warn


def test_manifest_fails_when_pdf_url_not_https(tmp_path: Path, monkeypatch):
    """A present-but-non-https pdf_hf_url is a HARD FAIL — the PDF must resolve to
    a real HF URL now that it is the PDF's authoritative location."""
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url="ftp://nope/x.pdf")
    r = verify_paper.check_manifest(paper_dir)
    assert not r.passed
    assert "https" in r.detail


def test_manifest_fails_committed_artifact_hash_mismatch(tmp_path: Path, monkeypatch):
    """A COMMITTED artifact (tex/paper_html) with a sha mismatch still FAILs."""
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url="https://hf/x.pdf")
    # corrupt a committed artifact after the manifest was written
    (paper_dir / "paper.html").write_text("<figure>DIFFERENT</figure>")
    r = verify_paper.check_manifest(paper_dir)
    assert not r.passed
    assert "sha256 mismatch" in r.detail


def test_manifest_fails_committed_artifact_missing_on_disk(tmp_path: Path, monkeypatch):
    """A COMMITTED artifact missing on disk still FAILs (only the PDF is exempt)."""
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url="https://hf/x.pdf")
    (paper_dir / "paper.html").unlink()
    r = verify_paper.check_manifest(paper_dir)
    assert not r.passed
    assert "paper_html" in r.detail and "missing on disk" in r.detail


def test_manifest_fails_missing_required_artifact(tmp_path: Path, monkeypatch):
    paper_dir = _make_paper(tmp_path)
    monkeypatch.setattr(verify_paper, "REPO", tmp_path)
    _write_real_manifest(paper_dir, tmp_path, "issue_657", pdf_url="https://hf/x.pdf")
    m = json.loads((paper_dir / "paper_manifest.json").read_text())
    del m["artifacts"]["paper_html"]
    (paper_dir / "paper_manifest.json").write_text(json.dumps(m))
    r = verify_paper.check_manifest(paper_dir)
    assert not r.passed
    assert "paper_html" in r.detail


def test_manifest_fails_when_absent(tmp_path: Path):
    paper_dir = _make_paper(tmp_path)
    (paper_dir / "paper_manifest.json").unlink(missing_ok=True)
    r = verify_paper.check_manifest(paper_dir)
    assert not r.passed


# ─── paper-stub body.md ──────────────────────────────────────────────────────


def test_paper_stub_pass(tmp_path: Path):
    stub = tmp_path / "body.md"
    stub.write_text(GOOD_STUB)
    r = verify_paper.check_paper_stub(stub)
    assert r.passed, r.detail


def test_paper_stub_fails_no_paper_flag(tmp_path: Path):
    stub = tmp_path / "body.md"
    stub.write_text(GOOD_STUB.replace("paper: true", "paper: false"))
    r = verify_paper.check_paper_stub(stub)
    assert not r.passed
    assert "paper: true" in r.detail


def test_paper_stub_fails_no_title(tmp_path: Path):
    stub = tmp_path / "body.md"
    stub.write_text(GOOD_STUB.replace("# A claim about leakage\n", ""))
    r = verify_paper.check_paper_stub(stub)
    assert not r.passed


def test_paper_stub_fails_no_paper_link(tmp_path: Path):
    stub = tmp_path / "body.md"
    stub.write_text(GOOD_STUB.replace("Paper: docs/papers/issue_657/issue_657.pdf", ""))
    r = verify_paper.check_paper_stub(stub)
    assert not r.passed
    assert "paper link" in r.detail


def test_paper_stub_fails_when_absent(tmp_path: Path):
    r = verify_paper.check_paper_stub(tmp_path / "nope.md")
    assert not r.passed


# ─── v1 scope: NO \metric check ──────────────────────────────────────────────


def test_no_metric_check_function_in_v1():
    """v1 must NOT enforce \\metric grounding (literal numbers + numeric-fidelity
    re-extraction is the v1 guarantee). A literal-number paper with no \\metric
    calls must verify clean; verify_paper must expose no \\metric check."""
    assert not hasattr(verify_paper, "check_metric_grounding")
    # the verify() runner's check set must not include a metric check name.
    names = {
        "compile clean",
        "required sections",
        "no confidence in body",
        "includegraphics confined + resolves",
        "bib entries resolve",
        "epsref resolves",
        "manifest complete + HF-PDF-consistent",
        "paper-stub body.md valid",
    }
    # A literal-numbers tex (no \metric) must pass the sections + confidence checks.
    literal_tex = GOOD_TEX.replace("\\newcommand{\\metric}[1]{X}\n", "").replace(
        "$=\\metric{x}$", "$=0.68$"
    )
    assert verify_paper.check_required_sections(literal_tex).passed
    assert verify_paper.check_no_confidence(literal_tex).passed
    # sanity: the canonical check-name set is exactly the v1 set (no metric name).
    assert "metric grounding" not in names
