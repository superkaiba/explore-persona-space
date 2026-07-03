"""Tests for scripts/verify_report.py — mechanical verifier for v2 report bodies.

Covers both modes (generation / promote) and each failure class the verifier
guards: structure/order, generation-time placeholders, promote-time filled
TLDR, the interpretive-lexicon scan (agent sections flagged; Motivation exempt),
image-file existence, and the optional planned-manifest coverage checks.

The report-v1-body-passes-set_clean_result invariant lives in
tests/test_task_workflow.py::test_set_clean_result_accepts_report_v1_body (that
file owns the git-backed fake_repo fixture set_clean_result needs).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_report.py"
_spec = importlib.util.spec_from_file_location("verify_report", _SCRIPT)
verify_report = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_report"] = verify_report
_spec.loader.exec_module(verify_report)  # type: ignore[union-attr]

PLACEHOLDER = verify_report.PLACEHOLDER


# ─── Body builders ──────────────────────────────────────────────────────────


def _default_sections(*, image: str = "figures/f.png") -> list[tuple[str, str]]:
    """The six required sections, in order, with a valid Results subsection."""
    return [
        ("## TLDR:", PLACEHOLDER),
        ("## Motivation:", "We hypothesize that base propensity predicts trained leakage."),
        (
            "## Methodology:",
            "We trained on 100 rows under two conditions: baseline and treatment.",
        ),
        ("## Metrics:", "Agreement rate (0-1), because it proxies the target behavior."),
        (
            "## Results:",
            "### rate by condition\n"
            "Bar chart of the agreement rate per condition.\n"
            f"![rate]({image})",
        ),
        ("## Next steps:", PLACEHOLDER),
    ]


def _assemble(
    sections: list[tuple[str, str]], *, title: str = "does X predict Y?", sentinel: bool = True
) -> str:
    lines = [f"# Experiment: {title}"]
    if sentinel:
        lines.append(verify_report.REPORT_SENTINEL)
    lines.append("")
    for header, content in sections:
        lines.append(header)
        lines.append(content)
        lines.append("")
    return "\n".join(lines) + "\n"


@pytest.fixture
def figs_root(tmp_path: Path) -> Path:
    """A figures-root with the default image present on disk."""
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "f.png").write_bytes(b"\x89PNG\r\n")
    return tmp_path


def _by_name(results, name):
    return next(r for r in results if r.name == name)


def _run(body: str, *, mode: str, figs_root: Path, manifest_path: Path | None = None):
    return verify_report.verify_report_text(
        body, mode=mode, figures_root=figs_root, manifest_path=manifest_path
    )


# ─── Happy paths ──────────────────────────────────────────────────────────


def test_generation_valid_passes(figs_root):
    ok, results = _run(_assemble(_default_sections()), mode="generation", figs_root=figs_root)
    assert ok, [r.render() for r in results if not r.passed]


def test_promote_valid_passes(figs_root):
    sections = _default_sections()
    sections[0] = ("## TLDR:", "Base propensity predicted trained leakage in the treatment arm.")
    sections[-1] = ("## Next steps:", "Run the ablation on more seeds.")
    ok, results = _run(_assemble(sections), mode="promote", figs_root=figs_root)
    assert ok, [r.render() for r in results if not r.passed]


# ─── Structural failures ────────────────────────────────────────────────


def test_wrong_order_fails(figs_root):
    sections = _default_sections()
    # Swap Motivation and Methodology.
    sections[1], sections[2] = sections[2], sections[1]
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "section-order").passed


def test_missing_section_fails(figs_root):
    sections = [s for s in _default_sections() if s[0] != "## Metrics:"]
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "required-sections").passed


def test_missing_sentinel_fails(figs_root):
    ok, results = _run(
        _assemble(_default_sections(), sentinel=False), mode="generation", figs_root=figs_root
    )
    assert not ok
    assert not _by_name(results, "sentinel").passed


def test_results_needs_exactly_one_image(figs_root):
    sections = _default_sections()
    sections[4] = ("## Results:", "### rate\nA description with no figure at all.")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "results-subsections").passed


# ─── Mode-specific TLDR / Next-steps ─────────────────────────────────────


def test_filled_tldr_at_generation_fails(figs_root):
    sections = _default_sections()
    sections[0] = ("## TLDR:", "A real takeaway written too early.")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "tldr-placeholder").passed


def test_empty_tldr_at_promote_fails(figs_root):
    sections = _default_sections()
    sections[0] = ("## TLDR:", "")  # empty at promote time
    ok, results = _run(_assemble(sections), mode="promote", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "tldr-filled").passed


def test_placeholder_tldr_at_promote_fails(figs_root):
    # Still the untouched placeholder → not filled → FAIL at promote.
    ok, results = _run(_assemble(_default_sections()), mode="promote", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "tldr-filled").passed


# ─── Interpretive-lexicon scan ───────────────────────────────────────────


def test_banned_lexeme_in_results_fails(figs_root):
    sections = _default_sections()
    sections[4] = (
        "## Results:",
        "### rate\nThis suggests the treatment worked.\n![rate](figures/f.png)",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "no-interpretive-lexicon").passed


def test_banned_lexeme_in_motivation_not_flagged(figs_root):
    sections = _default_sections()
    # Motivation is exempt — hypothesis framing ("suggests") is allowed there.
    sections[1] = ("## Motivation:", "Prior work suggests X predicts Y; we test whether it holds.")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert _by_name(results, "no-interpretive-lexicon").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_banned_lexeme_scanned_in_promote_mode_too(figs_root):
    # Agent sections are still lexicon-scanned at promote; only Thomas's
    # TLDR / Next-steps are exempt.
    sections = _default_sections()
    sections[0] = ("## TLDR:", "Thomas takeaway.")
    sections[2] = ("## Methodology:", "The design demonstrates that the method is sound.")
    ok, results = _run(_assemble(sections), mode="promote", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "no-interpretive-lexicon").passed


# ─── Image files ─────────────────────────────────────────────────────────


def test_missing_figure_file_fails(figs_root):
    sections = _default_sections(image="figures/missing.png")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "figure-files-exist").passed


# ─── Manifest coverage ───────────────────────────────────────────────────


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    p = tmp_path / "planned_manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def test_manifest_missing_condition_fails(figs_root, tmp_path):
    manifest = {
        "issue": 999,
        "conditions": ["baseline", "treatment", "phantom_condition"],
        "metrics": ["Agreement rate"],
        "figures": [],
    }
    mpath = _write_manifest(tmp_path, manifest)
    ok, results = _run(
        _assemble(_default_sections()), mode="generation", figs_root=figs_root, manifest_path=mpath
    )
    assert not ok
    cond = _by_name(results, "manifest-conditions")
    assert not cond.passed and "phantom_condition" in cond.detail


def test_manifest_figure_not_run_passes(figs_root, tmp_path):
    # A planned figure with no matching ### subsection is covered when the
    # report explicitly marks it "not run".
    sections = _default_sections()
    sections[3] = (
        "## Metrics:",
        "Agreement rate (0-1). The calibration curve figure was not run (insufficient data).",
    )
    manifest = {
        "issue": 999,
        "conditions": ["baseline", "treatment"],
        "metrics": ["Agreement rate"],
        "figures": [
            {
                "id": "calib",
                "title": "calibration curve",
                "source": "eval_results/issue_999/*.json",
                "transform": "reliability-binned mean, trained - base",
                "plotted_quantity": "observed vs predicted rate",
            }
        ],
    }
    mpath = _write_manifest(tmp_path, manifest)
    ok, results = _run(
        _assemble(sections), mode="generation", figs_root=figs_root, manifest_path=mpath
    )
    assert _by_name(results, "manifest-figures").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_manifest_figure_missing_fails(figs_root, tmp_path):
    # A planned figure neither plotted (### subsection) nor marked "not run".
    manifest = {
        "issue": 999,
        "conditions": ["baseline", "treatment"],
        "metrics": ["Agreement rate"],
        "figures": [
            {
                "id": "calib",
                "title": "calibration curve",
                "source": "eval_results/issue_999/*.json",
                "transform": "reliability-binned mean",
                "plotted_quantity": "observed vs predicted rate",
            }
        ],
    }
    mpath = _write_manifest(tmp_path, manifest)
    ok, results = _run(
        _assemble(_default_sections()), mode="generation", figs_root=figs_root, manifest_path=mpath
    )
    assert not ok
    figs = _by_name(results, "manifest-figures")
    assert not figs.passed and "calib" in figs.detail


def test_manifest_schema_invalid_fails(figs_root, tmp_path):
    # Missing the required "metrics" key.
    manifest = {"issue": 999, "conditions": ["baseline"], "figures": []}
    mpath = _write_manifest(tmp_path, manifest)
    ok, results = _run(
        _assemble(_default_sections()), mode="generation", figs_root=figs_root, manifest_path=mpath
    )
    assert not ok
    assert not _by_name(results, "manifest-schema").passed


# ─── htmlpreview SHA well-formedness ─────────────────────────────────────


def test_htmlpreview_missing_sha_fails(figs_root):
    sections = _default_sections()
    # An htmlpreview link pinned to a branch name, not a 40-hex SHA → FAIL.
    sections[2] = (
        "## Methodology:",
        "Dashboard: https://htmlpreview.github.io/?https://raw.githubusercontent.com/o/r/main/x.html",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "htmlpreview-sha").passed


def test_htmlpreview_valid_sha_passes(figs_root):
    sha = "a" * 40
    sections = _default_sections()
    sections[2] = (
        "## Methodology:",
        f"Dashboard: https://htmlpreview.github.io/?https://raw.githubusercontent.com/o/r/{sha}/x.html",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert _by_name(results, "htmlpreview-sha").passed
    assert ok, [r.render() for r in results if not r.passed]


# ─── CLI exit-code contract ──────────────────────────────────────────────


def test_cli_exit_codes(figs_root, tmp_path):
    good = tmp_path / "good.md"
    good.write_text(_assemble(_default_sections()))
    bad_sections = _default_sections()
    bad_sections[0] = ("## TLDR:", "filled too early")
    bad = tmp_path / "bad.md"
    bad.write_text(_assemble(bad_sections))

    r_ok = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--file",
            str(good),
            "--mode",
            "generation",
            "--figures-root",
            str(figs_root),
        ],
        capture_output=True,
        text=True,
    )
    assert r_ok.returncode == 0, r_ok.stdout + r_ok.stderr
    assert "OVERALL: PASS" in r_ok.stdout

    r_bad = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--file",
            str(bad),
            "--mode",
            "generation",
            "--figures-root",
            str(figs_root),
        ],
        capture_output=True,
        text=True,
    )
    assert r_bad.returncode == 1, r_bad.stdout + r_bad.stderr
    assert "OVERALL: FAIL" in r_bad.stdout


# ─── Verbatim content is DATA (fences + blockquotes blanked) ─────────────


def test_lexicon_ignores_blockquote_verbatim_in_methodology(figs_root):
    # A verbatim transcript quoted in Methodology carries interpretive lexemes
    # as DATA — a blockquote is blanked before the lexicon scan, so it PASSes.
    sections = _default_sections()
    sections[2] = (
        "## Methodology:",
        "We trained under two conditions. Verbatim transcript we showed the model:\n"
        "> User: is the claim true?\n"
        "> Assistant: This suggests it is, and demonstrates that the effect holds.",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert _by_name(results, "no-interpretive-lexicon").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_fenced_metrics_heading_does_not_shadow_real_metrics(figs_root):
    # A `## Metrics:` line inside a fenced code block must NOT be parsed as a
    # section heading and shadow the real Metrics section — the banned lexeme
    # in the REAL Metrics prose must still be flagged.
    sections = _default_sections()
    sections[2] = (
        "## Methodology:",
        "We trained under two conditions. Example config we pasted verbatim:\n"
        "```text\n"
        "## Metrics:\n"
        "an example metric line inside the fence\n"
        "```\n"
        "Training completed without error.",
    )
    sections[3] = ("## Metrics:", "Agreement rate. The design demonstrates that it is valid.")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    lex = _by_name(results, "no-interpretive-lexicon")
    assert not ok
    assert not lex.passed
    # The flagged lexeme comes from the REAL Metrics prose, not the fenced one.
    assert "Metrics" in lex.detail and "demonstrates that" in lex.detail


def test_image_existence_ignores_quoted_image_in_blockquote(figs_root):
    # A `![...](missing.png)` inside a blockquote is a verbatim reference to a
    # prior figure — blanked before the image-existence scan, so no FAIL.
    sections = _default_sections()
    sections[2] = (
        "## Methodology:",
        "We trained under two conditions. A figure from prior work we referenced:\n"
        "> ![old figure](figures/does_not_exist.png)",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert _by_name(results, "figure-files-exist").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_duplicate_required_section_fails(figs_root):
    # A second `## TLDR:` heading (also the intact placeholder, so the
    # placeholder check still passes) must trip the duplicate-section check.
    sections = _default_sections()
    sections.append(("## TLDR:", PLACEHOLDER))
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    dup = _by_name(results, "duplicate-section")
    assert not dup.passed and "## TLDR:" in dup.detail


def test_no_duplicate_section_on_valid_body(figs_root):
    ok, results = _run(_assemble(_default_sections()), mode="generation", figs_root=figs_root)
    assert _by_name(results, "duplicate-section").passed
    assert ok, [r.render() for r in results if not r.passed]


# ─── --issue resolves body.md via the task-workflow library ──────────────


@pytest.fixture
def issue_repo(tmp_path, monkeypatch):
    """Rebind task_workflow's path resolvers at a tmp repo so the REAL
    ``find_task_path`` resolves ``--issue N`` to ``tmp/tasks/<status>/N/body.md``.

    Mirrors the ``fake_repo`` fixture in tests/test_task_workflow.py: the
    2026-05-25 worktree-staleness fix replaced module constants with the
    ``repo_root()`` / ``tasks_dir()`` / ``registry_path()`` accessors, so tests
    monkeypatch the FUNCTIONS. verify_report imports ``find_task_path`` lazily
    inside ``main()``, picking up these patched resolvers.
    """
    (tmp_path / ".git").mkdir()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(tw, "registry_path", lambda: tmp_path / "tasks" / "REGISTRY.json")
    return tmp_path


def test_issue_resolves_body_via_library(issue_repo):
    repo = issue_repo
    task_dir = repo / "tasks" / "proposed" / "777"
    task_dir.mkdir(parents=True)
    (task_dir / "body.md").write_text(_assemble(_default_sections()))
    # figures-root defaults to the git-repo root of the resolved body.md.
    (repo / "figures").mkdir()
    (repo / "figures" / "f.png").write_bytes(b"\x89PNG\r\n")
    assert verify_report.main(["--issue", "777", "--mode", "generation"]) == 0


def test_issue_missing_task_returns_usage_error(issue_repo):
    # No such task on disk or in the registry → exit 2 (usage error), not crash.
    assert verify_report.main(["--issue", "424242", "--mode", "generation"]) == 2


def test_issue_and_file_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        verify_report.main(["--issue", "1", "--file", "x.md", "--mode", "generation"])


def test_neither_issue_nor_file_is_a_usage_error():
    with pytest.raises(SystemExit):
        verify_report.main(["--mode", "generation"])


# ─── Manifest matching: exact figure headings + word-boundary coverage ───


def test_manifest_figure_substring_heading_not_covered(figs_root, tmp_path):
    # The default Results subsection heading is "rate by condition"; a figure
    # titled "rate" is a SUBSTRING but not an exact match → NOT covered.
    manifest = {
        "issue": 999,
        "conditions": ["baseline", "treatment"],
        "metrics": ["Agreement rate"],
        "figures": [
            {
                "id": "ratefig",
                "title": "rate",
                "source": "eval_results/issue_999/*.json",
                "transform": "mean",
                "plotted_quantity": "rate per condition",
            }
        ],
    }
    mpath = _write_manifest(tmp_path, manifest)
    ok, results = _run(
        _assemble(_default_sections()), mode="generation", figs_root=figs_root, manifest_path=mpath
    )
    assert not ok
    figs = _by_name(results, "manifest-figures")
    assert not figs.passed and "ratefig" in figs.detail


def test_manifest_figure_exact_heading_covered(figs_root, tmp_path):
    # Exact (case-insensitive) match of the figure title to the ### heading.
    manifest = {
        "issue": 999,
        "conditions": ["baseline", "treatment"],
        "metrics": ["Agreement rate"],
        "figures": [
            {
                "id": "ratefig",
                "title": "Rate by Condition",  # exact-match (case-insensitive) of the ### heading
                "source": "eval_results/issue_999/*.json",
                "transform": "mean",
                "plotted_quantity": "rate per condition",
            }
        ],
    }
    mpath = _write_manifest(tmp_path, manifest)
    ok, results = _run(
        _assemble(_default_sections()), mode="generation", figs_root=figs_root, manifest_path=mpath
    )
    assert _by_name(results, "manifest-figures").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_manifest_condition_substring_not_covered(figs_root, tmp_path):
    # The default body mentions "leakage" but not the standalone word "leak";
    # a planned condition "leak" is a bare-substring hit but not a
    # word-boundary hit → NOT covered.
    manifest = {
        "issue": 999,
        "conditions": ["leak"],
        "metrics": ["Agreement rate"],
        "figures": [],
    }
    mpath = _write_manifest(tmp_path, manifest)
    ok, results = _run(
        _assemble(_default_sections()), mode="generation", figs_root=figs_root, manifest_path=mpath
    )
    assert not ok
    cond = _by_name(results, "manifest-conditions")
    assert not cond.passed and "leak" in cond.detail
