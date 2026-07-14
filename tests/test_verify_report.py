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

# The default Results image is a WELL-FORMED SHA pin (#1224 Option A: figures
# commit at Step 7b, before assembly, so a valid report always pins). The SHA
# is synthetic — in a non-git tmp figures-root the blob-identity check degrades
# to WARN (which counts as PASS overall).
_PIN_SHA = "a" * 40
_PINNED_IMAGE = f"https://raw.githubusercontent.com/o/r/{_PIN_SHA}/figures/issue_5/f.png"


def _pin(sha: str, path: str = "figures/issue_5/f.png") -> str:
    return f"https://raw.githubusercontent.com/o/r/{sha}/{path}"


# ─── Body builders ──────────────────────────────────────────────────────────


def _default_sections(*, image: str = _PINNED_IMAGE) -> list[tuple[str, str]]:
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


def _promote_sections(**kw) -> list[tuple[str, str]]:
    """Default sections with TLDR + Next steps filled (valid at promote time)."""
    sections = _default_sections(**kw)
    sections[0] = ("## TLDR:", "Thomas takeaway: the effect held.")
    sections[-1] = ("## Next steps:", "Run more seeds.")
    return sections


@pytest.fixture
def figs_root(tmp_path: Path) -> Path:
    """A figures-root with the default image present on disk."""
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "f.png").write_bytes(b"\x89PNG\r\n")
    return tmp_path


def _git_run(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.fixture
def git_figs_repo(tmp_path: Path) -> tuple[Path, str]:
    """A REAL git repo with figures/issue_5/f.png committed; returns (root, head_sha)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    _git_run(tmp_path, "config", "user.email", "test@test.test")
    _git_run(tmp_path, "config", "user.name", "Test")
    _git_run(tmp_path, "config", "commit.gpgsign", "false")
    figs = tmp_path / "figures" / "issue_5"
    figs.mkdir(parents=True)
    (figs / "f.png").write_bytes(b"\x89PNG\r\n\x1a\nreal-figure-bytes")
    _git_run(tmp_path, "add", "figures/issue_5/f.png")
    _git_run(tmp_path, "commit", "-q", "-m", "add figure")
    head = _git_run(tmp_path, "rev-parse", "HEAD")
    return tmp_path, head


def _by_name(results, name):
    return next(r for r in results if r.name == name)


def _run(
    body: str,
    *,
    mode: str,
    figs_root: Path,
    manifest_path: Path | None = None,
    expect_issue: int | None = None,
):
    return verify_report.verify_report_text(
        body,
        mode=mode,
        figures_root=figs_root,
        manifest_path=manifest_path,
        expect_issue=expect_issue,
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
        f"### rate\nThis suggests the treatment worked.\n![rate]({_PINNED_IMAGE})",
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
    # --issue 777 implies expect_issue=777, so the Results pin must name issue_777.
    body = _assemble(_default_sections(image=_pin(_PIN_SHA, "figures/issue_777/f.png")))
    (task_dir / "body.md").write_text(body)
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


# ─── image-pin-format (#1224 mechanization) ───────────────────────────────


def test_pinned_image_wellformed_passes(figs_root):
    ok, results = _run(_assemble(_default_sections()), mode="generation", figs_root=figs_root)
    fmt = _by_name(results, "image-pin-format")
    ident = _by_name(results, "image-pin-blob-identity")
    assert fmt.passed and not fmt.is_warn
    assert ident.passed and ident.is_warn  # non-git tmp figures-root → WARN (counts as PASS)
    assert ok, [r.render() for r in results if not r.passed]


def test_unpinned_relative_results_image_fails(figs_root):
    # A relative-path Results image violates the pin contract in BOTH modes.
    for mode, builder in (("generation", _default_sections), ("promote", _promote_sections)):
        ok, results = _run(
            _assemble(builder(image="figures/f.png")), mode=mode, figs_root=figs_root
        )
        assert not ok
        assert not _by_name(results, "image-pin-format").passed


def test_main_ref_image_fails(figs_root):
    sections = _default_sections(
        image="https://raw.githubusercontent.com/o/r/main/figures/issue_5/f.png"
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "image-pin-format").passed


def test_non_figures_path_pin_fails(figs_root):
    sections = _default_sections(image=_pin(_PIN_SHA, "docs/x.png"))
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "image-pin-format").passed


def test_expect_issue_mismatch_fails(figs_root):
    sections = _default_sections(image=_pin(_PIN_SHA, "figures/issue_7/f.png"))
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root, expect_issue=5)
    assert not ok
    fmt = _by_name(results, "image-pin-format")
    assert not fmt.passed and "expected issue 5" in fmt.detail


def test_outside_results_raw_image_exempt_from_issue_match(figs_root):
    # A well-formed CROSS-ISSUE pin outside Results with expect_issue=5:
    # the issue-number match is Results-scoped → NO format FAIL.
    sections = _default_sections()
    sections[2] = (
        "## Methodology:",
        "We trained under two conditions. Prior figure: "
        f"![prior]({_pin('c' * 40, 'figures/issue_9/x.png')})",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root, expect_issue=5)
    assert _by_name(results, "image-pin-format").passed
    assert ok, [r.render() for r in results if not r.passed]
    # A MALFORMED raw URL outside Results (short SHA) still fails well-formedness.
    sections[2] = (
        "## Methodology:",
        "Prior figure: "
        "![prior](https://raw.githubusercontent.com/o/r/abc123/figures/issue_9/x.png)",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root, expect_issue=5)
    assert not ok
    assert not _by_name(results, "image-pin-format").passed


# ─── image-pin-blob-identity (#1224 mechanization) ────────────────────────


def test_pin_blob_identity_match_passes(git_figs_repo):
    repo, head = git_figs_repo
    ok, results = _run(
        _assemble(_default_sections(image=_pin(head))), mode="generation", figs_root=repo
    )
    ident = _by_name(results, "image-pin-blob-identity")
    assert ident.passed and not ident.is_warn
    assert ok, [r.render() for r in results if not r.passed]


def test_pin_blob_identity_mismatch_fails_generation(git_figs_repo):
    # At 7e the worktree copy IS the just-plotted figure; a pin whose blob
    # differs is the exact wrong-SHA bug class → generation FAIL.
    repo, head = git_figs_repo
    (repo / "figures" / "issue_5" / "f.png").write_bytes(b"\x89PNG modified-after-commit")
    ok, results = _run(
        _assemble(_default_sections(image=_pin(head))), mode="generation", figs_root=repo
    )
    assert not ok
    ident = _by_name(results, "image-pin-blob-identity")
    assert not ident.passed and "differs" in ident.detail


def test_pin_blob_identity_mismatch_warns_promote(git_figs_repo):
    # Post-merge local drift is a stray; the pin is the record (#922) → WARN.
    repo, head = git_figs_repo
    (repo / "figures" / "issue_5" / "f.png").write_bytes(b"\x89PNG modified-after-commit")
    ok, results = _run(
        _assemble(_promote_sections(image=_pin(head))), mode="promote", figs_root=repo
    )
    ident = _by_name(results, "image-pin-blob-identity")
    assert ident.passed and ident.is_warn
    assert ok, [r.render() for r in results if not r.passed]


def test_pin_commit_lacks_path_fails(git_figs_repo):
    # Commit resolves but does not contain the pinned path → a wrong pin,
    # definitively — FAIL in BOTH modes.
    repo, head = git_figs_repo
    ghost = _pin(head, "figures/issue_5/ghost.png")
    ok, results = _run(_assemble(_default_sections(image=ghost)), mode="generation", figs_root=repo)
    assert not ok
    ident = _by_name(results, "image-pin-blob-identity")
    assert not ident.passed and "does not contain" in ident.detail
    ok_p, results_p = _run(
        _assemble(_promote_sections(image=ghost)), mode="promote", figs_root=repo
    )
    assert not ok_p
    assert not _by_name(results_p, "image-pin-blob-identity").passed


def test_pin_unresolvable_sha_fails_generation_warns_promote(git_figs_repo):
    # At 7e the pin commit was JUST created locally, so an unresolvable SHA is
    # the fabricated/hallucinated-SHA class → generation FAIL; at promote an
    # unfetched clone is plausible → WARN.
    repo, _head = git_figs_repo
    ok, results = _run(
        _assemble(_default_sections(image=_pin("b" * 40))), mode="generation", figs_root=repo
    )
    assert not ok
    ident = _by_name(results, "image-pin-blob-identity")
    assert not ident.passed and "unresolvable" in ident.detail
    ok_p, results_p = _run(
        _assemble(_promote_sections(image=_pin("b" * 40))), mode="promote", figs_root=repo
    )
    ident_p = _by_name(results_p, "image-pin-blob-identity")
    assert ident_p.passed and ident_p.is_warn
    assert ok_p, [r.render() for r in results_p if not r.passed]


def test_non_git_checkout_degrades_to_warn(figs_root):
    ok, results = _run(_assemble(_default_sections()), mode="generation", figs_root=figs_root)
    ident = _by_name(results, "image-pin-blob-identity")
    assert ident.passed and ident.is_warn and "not a git checkout" in ident.detail
    assert ok, [r.render() for r in results if not r.passed]


def test_mixed_sha_results_pins_both_valid_pass(git_figs_repo):
    # Two Results pins at DIFFERENT real SHAs (the 7b re-entry /
    # partial-re-splice shape): each pin verifies independently; only the
    # issue NUMBER must be identical across Results images, never the SHA.
    repo, sha1 = git_figs_repo
    (repo / "figures" / "issue_5" / "g.png").write_bytes(b"\x89PNG second-figure-bytes")
    _git_run(repo, "add", "figures/issue_5/g.png")
    _git_run(repo, "commit", "-q", "-m", "add second figure")
    sha2 = _git_run(repo, "rev-parse", "HEAD")
    assert sha1 != sha2
    sections = _default_sections()
    sections[4] = (
        "## Results:",
        "### rate by condition\n"
        "Bar chart of the agreement rate per condition.\n"
        f"![rate]({_pin(sha1)})\n"
        "### rate per unit\n"
        "Per-unit points behind the aggregate.\n"
        f"![points]({_pin(sha2, 'figures/issue_5/g.png')})",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=repo)
    assert _by_name(results, "image-pin-format").passed
    ident = _by_name(results, "image-pin-blob-identity")
    assert ident.passed and not ident.is_warn
    assert ok, [r.render() for r in results if not r.passed]


def test_pin_resolves_no_local_copy_warns_generation(git_figs_repo):
    # Pin resolves in the object DB but the local copy is gone: at 7e every
    # Results figure should have a just-plotted local copy → generation WARN;
    # post-merge that is expected → promote PASS-note.
    repo, head = git_figs_repo
    (repo / "figures" / "issue_5" / "f.png").unlink()
    ok, results = _run(
        _assemble(_default_sections(image=_pin(head))), mode="generation", figs_root=repo
    )
    ident = _by_name(results, "image-pin-blob-identity")
    assert ident.passed and ident.is_warn and "no local copy" in ident.detail
    assert ok, [r.render() for r in results if not r.passed]
    ok_p, results_p = _run(
        _assemble(_promote_sections(image=_pin(head))), mode="promote", figs_root=repo
    )
    ident_p = _by_name(results_p, "image-pin-blob-identity")
    assert ident_p.passed and not ident_p.is_warn and "no local copy" in ident_p.detail
    assert ok_p, [r.render() for r in results_p if not r.passed]


def test_cli_expect_issue_flag(figs_root, tmp_path):
    good = tmp_path / "good.md"
    good.write_text(_assemble(_default_sections()))
    base = [
        sys.executable,
        str(_SCRIPT),
        "--file",
        str(good),
        "--mode",
        "generation",
        "--figures-root",
        str(figs_root),
    ]
    r_ok = subprocess.run([*base, "--expect-issue", "5"], capture_output=True, text=True)
    assert r_ok.returncode == 0, r_ok.stdout + r_ok.stderr
    r_mismatch = subprocess.run([*base, "--expect-issue", "7"], capture_output=True, text=True)
    assert r_mismatch.returncode == 1, r_mismatch.stdout + r_mismatch.stderr
    assert "image-pin-format" in r_mismatch.stdout
    # --issue + --expect-issue together is an argparse usage error (exit 2):
    # silently ignoring one of them would be a footgun.
    with pytest.raises(SystemExit) as exc:
        verify_report.main(["--issue", "1", "--expect-issue", "5", "--mode", "generation"])
    assert exc.value.code == 2
