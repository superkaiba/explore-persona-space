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


def _results_block(image: str, *, takeaways: str = PLACEHOLDER) -> str:
    return (
        "### rate by condition\n"
        "Bar chart of the agreement rate per condition.\n"
        "**Methodology**\n"
        "- Agreement rate per condition; x-axis: condition, y-axis: rate (0-1); n = 100.\n"
        f"![rate]({image})\n"
        "**Takeaways**\n"
        f"{takeaways}"
    )


def _default_sections(*, image: str = _PINNED_IMAGE) -> list[tuple[str, str]]:
    """The five required sections, in order, with a valid Results subsection.

    Shared metrics live INSIDE ``## Methodology (shared)`` (the official
    template folds the former ``## Metrics:`` H2 into a ``**Metrics:**``
    block); each Results subsection carries its own ``**Methodology**`` block.
    """
    return [
        ("## Motivation", "We hypothesize that base propensity predicts trained leakage."),
        ("## TLDR", PLACEHOLDER),
        (
            "## Methodology (shared)",
            "We trained on 100 rows under two conditions: baseline and treatment.\n"
            "- **Metrics:** Agreement rate (0-1), because it proxies the target behavior.",
        ),
        ("## Results", _results_block(image)),
        ("## Conclusion and next steps", PLACEHOLDER),
    ]


# The body's detailed-companion link (two-document output). The default pins
# the same synthetic SHA as the default image; well-formedness-only check.
def _detailed_link(issue: int = 5, sha: str = _PIN_SHA) -> str:
    return (
        "**Detailed writeup:** "
        f"https://github.com/o/r/blob/{sha}/docs/reports/issue_{issue}_detailed.md"
    )


def _assemble(
    sections: list[tuple[str, str]],
    *,
    title: str = "does X predict Y?",
    sentinel: bool = True,
    h1_prefix: str = "Experiment: ",
    detailed_link: str | None = _detailed_link(),
) -> str:
    lines = [f"# {h1_prefix}{title}"]
    if sentinel:
        lines.append(verify_report.REPORT_SENTINEL)
    lines.append("")
    if detailed_link is not None:
        lines.append(detailed_link)
        lines.append("")
    for header, content in sections:
        lines.append(header)
        lines.append(content)
        lines.append("")
    return "\n".join(lines) + "\n"


def _promote_sections(**kw) -> list[tuple[str, str]]:
    """Default sections with TLDR + Conclusion filled (valid at promote time)."""
    sections = _default_sections(**kw)
    sections[1] = ("## TLDR", "Thomas takeaway: the effect held.")
    sections[-1] = ("## Conclusion and next steps", "Run more seeds.")
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
    sections[1] = ("## TLDR", "Base propensity predicted trained leakage in the treatment arm.")
    sections[-1] = ("## Conclusion and next steps", "Run the ablation on more seeds.")
    ok, results = _run(_assemble(sections), mode="promote", figs_root=figs_root)
    assert ok, [r.render() for r in results if not r.passed]


def test_result_title_accepted_at_promote_only(figs_root):
    # Thomas retitles the H1 to `# Result: <claim>` at TLDR time: promote
    # accepts it; at generation the claim form violates interpretivity.
    body = _assemble(_promote_sections(), title="X predicts Y", h1_prefix="Result: ")
    ok, results = _run(body, mode="promote", figs_root=figs_root)
    assert _by_name(results, "h1-title").passed
    assert ok, [r.render() for r in results if not r.passed]
    gen_body = _assemble(_default_sections(), title="X predicts Y", h1_prefix="Result: ")
    ok_g, results_g = _run(gen_body, mode="generation", figs_root=figs_root)
    assert not ok_g
    assert not _by_name(results_g, "h1-title").passed


def test_headings_with_trailing_colons_pass(figs_root):
    # The canonical headings carry no trailing colon; the verifier normalizes
    # heading lines, so the colon-suffixed forms are accepted too.
    sections = [(h + ":", c) for h, c in _default_sections()]
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert ok, [r.render() for r in results if not r.passed]


def test_grandfathered_old_section_names_pass(figs_root):
    # Pre-2026-07-30 H2 names (`## Methodology:` / `## Next steps:`) normalize
    # to the canonical `## Methodology (shared)` / `## Conclusion and next
    # steps`, so old bodies keep verifying — including the placeholder check
    # reading the aliased Conclusion section.
    sections = _default_sections()
    sections[2] = ("## Methodology:", sections[2][1])
    sections[4] = ("## Next steps:", PLACEHOLDER)
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert _by_name(results, "required-sections").passed
    assert _by_name(results, "conclusion-placeholder").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_grandfathered_old_results_shape_warns_at_promote(figs_root):
    # A full pre-2026-07-30 body: old H2 names, a `**Plot:**` label,
    # `**Takeaways:**` with a trailing colon, and NO per-result
    # `**Methodology**` block — promote mode WARNs (counts as PASS);
    # generation mode requires the per-result Methodology block and FAILs.
    old_results = (
        "### rate by condition\n"
        "Bar chart of the agreement rate per condition.\n"
        "**Plot: rate by condition**\n"
        f"![rate]({_PINNED_IMAGE})\n"
        "**Takeaways:**\n"
        "- The effect held: rate 0.7 vs 0.2."
    )
    sections = [
        ("## Motivation:", "We test whether base propensity predicts leakage."),
        ("## TLDR:", "Thomas takeaway: the effect held."),
        ("## Methodology:", "We trained on 100 rows.\n- **Metrics:** Agreement rate (0-1)."),
        ("## Results:", old_results),
        ("## Next steps:", "Run more seeds."),
    ]
    body = _assemble(sections, title="X predicts Y", h1_prefix="Result: ", detailed_link=None)
    ok, results = _run(body, mode="promote", figs_root=figs_root)
    sub = _by_name(results, "results-subsections")
    assert sub.passed and sub.is_warn and "Methodology" in sub.detail
    # A grandfathered body has no detailed-companion link either → WARN only.
    link = _by_name(results, "detailed-writeup-link")
    assert link.passed and link.is_warn
    assert ok, [r.render() for r in results if not r.passed]


def test_missing_per_result_methodology_fails_generation(figs_root):
    sections = _default_sections()
    sections[3] = (
        "## Results",
        "### rate\nBar chart of the agreement rate per condition.\n"
        f"![rate]({_PINNED_IMAGE})\n**Takeaways**\n{PLACEHOLDER}",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    sub = _by_name(results, "results-subsections")
    assert not sub.passed and "**Methodology**" in sub.detail


def test_takeaways_trailing_colon_form_accepted(figs_root):
    # The grandfathered `**Takeaways:**` label is accepted in both modes.
    sections = _default_sections()
    sections[3] = (
        "## Results",
        _results_block(_PINNED_IMAGE).replace("**Takeaways**", "**Takeaways:**"),
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert ok, [r.render() for r in results if not r.passed]


def test_missing_detailed_link_fails_generation(figs_root):
    # The summarized body must link its detailed companion at generation time.
    ok, results = _run(
        _assemble(_default_sections(), detailed_link=None), mode="generation", figs_root=figs_root
    )
    assert not ok
    link = _by_name(results, "detailed-writeup-link")
    assert not link.passed and "Detailed writeup" in link.detail


def test_malformed_detailed_link_fails_both_modes(figs_root):
    # Branch-pinned (not 40-hex) detailed link → FAIL in both modes.
    bad = "**Detailed writeup:** https://github.com/o/r/blob/main/docs/reports/issue_5_detailed.md"
    for mode, builder in (("generation", _default_sections), ("promote", _promote_sections)):
        ok, results = _run(_assemble(builder(), detailed_link=bad), mode=mode, figs_root=figs_root)
        assert not ok
        assert not _by_name(results, "detailed-writeup-link").passed


def test_detailed_link_issue_mismatch_fails(figs_root):
    ok, results = _run(
        _assemble(_default_sections(), detailed_link=_detailed_link(9)),
        mode="generation",
        figs_root=figs_root,
        expect_issue=5,
    )
    assert not ok
    link = _by_name(results, "detailed-writeup-link")
    assert not link.passed and "expected issue 5" in link.detail


def test_detailed_link_angle_bracket_form_accepted(figs_root):
    # The template skeleton displays the URL slot as `<https://...>`; the
    # angle-bracket-wrapped form is accepted (stripped before matching).
    wrapped = (
        "**Detailed writeup:** "
        f"<https://github.com/o/r/blob/{_PIN_SHA}/docs/reports/issue_5_detailed.md>"
    )
    ok, results = _run(
        _assemble(_default_sections(), detailed_link=wrapped),
        mode="generation",
        figs_root=figs_root,
    )
    assert _by_name(results, "detailed-writeup-link").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_duplicate_detailed_link_fails(figs_root):
    # A follow-up re-pin must REPLACE the old line — two link lines FAIL.
    doubled = _detailed_link() + "\n\n" + _detailed_link(5, "b" * 40)
    ok, results = _run(
        _assemble(_default_sections(), detailed_link=doubled),
        mode="generation",
        figs_root=figs_root,
    )
    assert not ok
    link = _by_name(results, "detailed-writeup-link")
    assert not link.passed and "exactly one" in link.detail


def test_detailed_link_raw_url_form_accepted(figs_root):
    raw = (
        "**Detailed writeup:** "
        f"https://raw.githubusercontent.com/o/r/{_PIN_SHA}/docs/reports/issue_5_detailed.md"
    )
    ok, results = _run(
        _assemble(_default_sections(), detailed_link=raw), mode="generation", figs_root=figs_root
    )
    assert _by_name(results, "detailed-writeup-link").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_retired_plot_label_fails_generation(figs_root):
    # A freshly assembled report must not carry the retired `**Plot:**` label
    # (promote tolerates it — see the grandfathered-shape test above).
    sections = _default_sections()
    sections[3] = (
        "## Results",
        "### rate by condition\n"
        "Bar chart of the agreement rate per condition.\n"
        "**Methodology**\n"
        "- Agreement rate per condition; n = 100.\n"
        "**Plot: rate by condition**\n"
        f"![rate]({_PINNED_IMAGE})\n"
        f"**Takeaways**\n{PLACEHOLDER}",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    sub = _by_name(results, "results-subsections")
    assert not sub.passed and "retired" in sub.detail


# ─── Structural failures ────────────────────────────────────────────────


def test_wrong_order_fails(figs_root):
    sections = _default_sections()
    # Swap Motivation and Methodology.
    sections[0], sections[2] = sections[2], sections[0]
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "section-order").passed


def test_missing_section_fails(figs_root):
    sections = [s for s in _default_sections() if s[0] != "## Methodology (shared)"]
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "required-sections").passed


def test_separate_metrics_section_is_not_required(figs_root):
    # The former `## Metrics:` H2 is retired — a report WITHOUT it passes
    # (metrics live inside Methodology), and adding one back does not shadow
    # any required section.
    ok, results = _run(_assemble(_default_sections()), mode="generation", figs_root=figs_root)
    assert _by_name(results, "required-sections").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_missing_sentinel_fails(figs_root):
    ok, results = _run(
        _assemble(_default_sections(), sentinel=False), mode="generation", figs_root=figs_root
    )
    assert not ok
    assert not _by_name(results, "sentinel").passed


def test_results_needs_exactly_one_image(figs_root):
    sections = _default_sections()
    sections[3] = (
        "## Results:",
        f"### rate\nA description with no figure at all.\n**Takeaways:**\n{PLACEHOLDER}",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "results-subsections").passed


def test_results_needs_takeaways_block(figs_root):
    # Every Results subsection carries a **Takeaways:** block (Thomas's slot).
    sections = _default_sections()
    sections[3] = (
        "## Results:",
        f"### rate\nBar chart of the agreement rate per condition.\n![rate]({_PINNED_IMAGE})",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    sub = _by_name(results, "results-subsections")
    assert not sub.passed and "Takeaways" in sub.detail


def test_agent_filled_takeaways_at_generation_fails(figs_root):
    # At generation the Takeaways content must be the intact placeholder —
    # an agent-written claim under a plot violates interpretivity.
    sections = _default_sections(image=_PINNED_IMAGE)
    sections[3] = ("## Results:", _results_block(_PINNED_IMAGE, takeaways="The effect held."))
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    sub = _by_name(results, "results-subsections")
    assert not sub.passed and "placeholder" in sub.detail


def test_filled_takeaways_at_promote_passes(figs_root):
    sections = _promote_sections()
    sections[3] = (
        "## Results:",
        _results_block(_PINNED_IMAGE, takeaways="- The mapping exists: R^2 0.705 [0.691, 0.719]."),
    )
    ok, results = _run(_assemble(sections), mode="promote", figs_root=figs_root)
    assert ok, [r.render() for r in results if not r.passed]


# ─── Mode-specific TLDR / Next-steps ─────────────────────────────────────


def test_filled_tldr_at_generation_fails(figs_root):
    sections = _default_sections()
    sections[1] = ("## TLDR:", "A real takeaway written too early.")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "tldr-placeholder").passed


def test_empty_tldr_at_promote_fails(figs_root):
    sections = _default_sections()
    sections[1] = ("## TLDR:", "")  # empty at promote time
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
    sections[3] = (
        "## Results:",
        "### rate\nThis suggests the treatment worked.\n"
        f"![rate]({_PINNED_IMAGE})\n**Takeaways:**\n{PLACEHOLDER}",
    )
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "no-interpretive-lexicon").passed


def test_banned_lexeme_in_motivation_not_flagged(figs_root):
    sections = _default_sections()
    # Motivation is exempt — hypothesis framing ("suggests") is allowed there.
    sections[0] = ("## Motivation:", "Prior work suggests X predicts Y; we test whether it holds.")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    assert _by_name(results, "no-interpretive-lexicon").passed
    assert ok, [r.render() for r in results if not r.passed]


def test_banned_lexeme_scanned_in_promote_mode_too(figs_root):
    # Methodology (pure agent prose) is still lexicon-scanned at promote;
    # Thomas's TLDR / Next-steps / Results takeaways are exempt.
    sections = _default_sections()
    sections[1] = ("## TLDR:", "Thomas takeaway.")
    sections[2] = ("## Methodology:", "The design demonstrates that the method is sound.")
    ok, results = _run(_assemble(sections), mode="promote", figs_root=figs_root)
    assert not ok
    assert not _by_name(results, "no-interpretive-lexicon").passed


def test_results_lexeme_not_flagged_at_promote(figs_root):
    # At promote the Results section carries Thomas's filled Takeaways +
    # claim-shaped headings — his voice is never lexicon-checked.
    sections = _promote_sections()
    sections[3] = (
        "## Results:",
        "### Result 1: the treatment worked\n"
        "Bar chart of the agreement rate per condition.\n"
        f"![rate]({_PINNED_IMAGE})\n**Takeaways:**\n"
        "- This suggests the treatment worked and confirms the hypothesis.",
    )
    ok, results = _run(_assemble(sections), mode="promote", figs_root=figs_root)
    assert _by_name(results, "no-interpretive-lexicon").passed
    assert ok, [r.render() for r in results if not r.passed]


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
    sections[2] = (
        "## Methodology:",
        "We trained on 100 rows under two conditions: baseline and treatment.\n"
        "- **Metrics:** Agreement rate (0-1). The calibration curve figure was "
        "not run (insufficient data).",
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
    bad_sections[1] = ("## TLDR:", "filled too early")
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


def test_fenced_heading_does_not_shadow_real_section(figs_root):
    # A `## Methodology:` line inside a fenced code block must NOT be parsed as
    # a section heading and shadow the real Methodology section — the banned
    # lexeme in the REAL Methodology prose must still be flagged.
    sections = _default_sections()
    sections[0] = (
        "## Motivation:",
        "We test whether X predicts Y. Example config we pasted verbatim:\n"
        "```text\n"
        "## Methodology:\n"
        "an example line inside the fence\n"
        "```\n"
        "The prior experiment used the same recipe.",
    )
    sections[2] = ("## Methodology:", "Agreement rate. The design demonstrates that it is valid.")
    ok, results = _run(_assemble(sections), mode="generation", figs_root=figs_root)
    lex = _by_name(results, "no-interpretive-lexicon")
    assert not ok
    assert not lex.passed
    # The flagged lexeme comes from the REAL Methodology prose, not the fenced one.
    assert "Methodology" in lex.detail and "demonstrates that" in lex.detail


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
    assert not dup.passed and "## TLDR" in dup.detail


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
    body = _assemble(
        _default_sections(image=_pin(_PIN_SHA, "figures/issue_777/f.png")),
        detailed_link=_detailed_link(777),
    )
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
    sections[3] = (
        "## Results",
        "### rate by condition\n"
        "Bar chart of the agreement rate per condition.\n"
        "**Methodology**\n"
        "- Agreement rate per condition; n = 100.\n"
        f"![rate]({_pin(sha1)})\n"
        f"**Takeaways**\n{PLACEHOLDER}\n"
        "### rate per unit\n"
        "Per-unit points behind the aggregate.\n"
        "**Methodology**\n"
        "- Per-unit points behind the aggregate; n = 100.\n"
        f"![points]({_pin(sha2, 'figures/issue_5/g.png')})\n"
        f"**Takeaways**\n{PLACEHOLDER}",
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


# ─── #2191: committed-under-claims + code-sha-cards ─────────────────────────
#
# Frozen #2162 fixture excerpts. tasks/running/2162/artifacts/
# issue-2162-report-sections.md is UNTRACKED (no git history) and was
# live-edited during #2191's own planning (it changed twice; origin/issue-2162
# advanced db5d1680a2 -> 434c84f5ae mid-plan), so tests NEVER read the live
# file — the lines below were frozen verbatim at implementation time
# (2026-08-08). The plan's §2 ground-truth measurements (blob counts, the
# usable-card SHA set) are the invariants, not line numbers.

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_resolves(token: str) -> bool:
    """Whether ``git rev-parse --verify <token>`` succeeds in the repo under test."""
    return (
        subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--verify", token],
            capture_output=True,
        ).returncode
        == 0
    )


requires_2162_pin = pytest.mark.skipif(
    not _repo_resolves("20fcef9c28^{commit}"),
    reason="#2162 branch pin 20fcef9c28 not in the local object DB (sparse/unfetched clone)",
)
requires_2162_branch = pytest.mark.skipif(
    not _repo_resolves("origin/issue-2162^{commit}"),
    reason="origin/issue-2162 unfetched in the repo under test",
)

_B4AB = "b4ab6ed5f96216566b78b090f432d763246997b0"
_BA34 = "ba3485b619e9d8b35dad58d9c4746511b59f5d28"
_EC11 = "ec113fdc05daecbfa5e04a7740552ed1093f079b"

_2162_L28 = (
    "  - **Stage 1 (primary, confirmatory):** full-state replace at ALL 28 layers at the slot — "
    "the maximal single-position intervention (Source: plan §4.2; #2094's largest clean effect "
    "at this exact cell). Grid decoding: temperature 1.0, K=5 draws per pair \u00d7 arm "
    "(`GRID_TEMPERATURE = 1.0`, `GRID_DRAWS = 5` in `scripts/issue2162_run.py` @ "
    f"`{_B4AB}`, the grid/anchors-phase commit); anchors K=10 at "
    "temperature 1.0 (`ANCHOR_DRAWS = 10`)."
)
_2162_L37 = (
    "  - P6 judge outputs: 748 files / 223 MB persisted on HF "
    "(`issue2162_ctxinfo/raw_completions/judge_raw/`; run marker). In git under "
    "`eval_results/issue_2162/judge/` at the branch pin (`20fcef9c28…`): "
    "`judge_summary.json`, `pools.json`, 7 gate reports, 70 raw files, the `anchors` + "
    "`stage2` audits (the grid audit is untracked), and 1 of 168 items files "
    "(`coherence.anchors`); the per-wave scores/items corpus (336 scores files on disk) is "
    "NOT committed — it lives on the HF data repo under `raw_completions/judge_raw/` per the "
    "wave-output convention."
)
_2162_L83 = (
    "  | Model | `Qwen/Qwen2.5-7B-Instruct` (bf16, 28 layers, H=3584) | repro card, "
    "`upload_done.json`; `MODEL_ID`/`HIDDEN_FULL`/`N_MODEL_LAYERS_FULL`, `issue2162_run.py` @ "
    "`b4ab6ed5f9…` |"
)
_2162_L88 = (
    "  | Stage-1 intervention | full-state replace, all 28 layers, one slot | plan §4.2; "
    "`joint_hooks(model, list(layers))` over all 28 layers, `issue2162_run.py` @ "
    "`b4ab6ed5f9…` (reuses `issue2094/hooks.py` unmodified) |"
)
_2162_L90 = (
    "  | Grid decoding | temperature 1.0, K=5 draws/pair\u00d7arm | "
    "`GRID_TEMPERATURE`/`GRID_DRAWS`, "
    "`issue2162_run.py` @ `b4ab6ed5f9…`; repro card |"
)
_2162_L111 = (
    f"  | Code SHAs | stage-1 grid/anchors `{_B4AB}` · margin `{_BA34}` · stage-2 `{_EC11}` · "
    "analysis outputs at consolidation commit `b228639eace6ebbdb65a2ef36f55f48684e01f4b` "
    "(ancestor of the branch pin `20fcef9c28…`, `issue-2162`) | grid/anchors: "
    "`repro.git_commit` in `gates/pilot_gate_report.json` + "
    "`judge/gates/separation_gate_report.json`; margin: reproducibility card + "
    "`final_commit_sha` in `margin/upload_done.json`; stage-2: launch marker; "
    "`git rev-parse` |"
)
_2162_L116 = (
    "- **Artifacts index:** rollout text `issue2162_ctxinfo/raw_completions/{anchors (16 "
    "shards), grid (234), stage2 (140), judge_raw, anchors_gate}` and tensors/manifests "
    "`issue2162_ctxinfo/analysis_tensors/{vc_bank (incl. bank.json + P1 gate reports), "
    "va_store, margin, probe_perm_matrix, manifests}` on `superkaiba1/explore-persona-space-"
    "data` (revision-pinned: https://huggingface.co/datasets/superkaiba1/explore-persona-"
    "space-data/tree/dc8108ab84f33695bbc769da0e6e8e2327f51eeb/issue2162_ctxinfo — the repo "
    "tip at authoring time; every `issue2162_ctxinfo` artifact cited here resolves at this "
    "revision); per-cell tables + gate reports in git under "
    "`eval_results/issue_2162/{f_metrics, judge, gates, margin, stage2}` on branch "
    "`issue-2162` (pushed; tip includes `b228639eac…`); the committed `judge/` subtree is "
    "the summary/pools/gates/raw/audits set enumerated under Realized run counts — per-wave "
    "judge scores/items are HF-side (`raw_completions/judge_raw/`), not in git."
)
_2162_CORRECTED_LINES = [
    _2162_L28,
    _2162_L37,
    _2162_L83,
    _2162_L88,
    _2162_L90,
    _2162_L111,
    _2162_L116,
]


def _round1_reconstruction() -> str:
    """The #2162 round-1 defect per CORRECTION 2 of the round-1
    epm:methodology-check verdict: the draft cited ``ba3485b619…`` as "the
    code" THROUGHOUT — ``b4ab6ed5f9…`` appeared nowhere — so every b4ab
    citation (full-hex and ellipsis-abbreviated) reverts to the ba34 form."""
    text = "\n".join(_2162_CORRECTED_LINES)
    return text.replace(_B4AB, _BA34).replace("b4ab6ed5f9", "ba3485b619")


# ── Integration tests (skipif-gated on the real object DB; the hermetic
#    tmp-repo twins below carry the logic coverage on sparse/unfetched clones).


@requires_2162_pin
def test_committed_under_fires_on_2162_round1():
    """Criterion 1 — check (a) FIRES on the EMPTY-PATH FORM of the #2162
    round-1 error class.

    Fixture provenance — and what this fixture is NOT: the round-1 draft is
    NOT recoverable verbatim (issue-2162-report-sections.md is untracked with
    no git history and was corrected in place); the authoritative description
    is MUST-FIX item 3 of the round-1 epm:methodology-check verdict
    (tasks/running/2162/events.jsonl, marker ts 2026-08-08T02:18:52Z): at
    ``20fcef9c28`` there are ZERO files under ``judge/scores/`` (336 on disk,
    all untracked) while the parent ``judge/`` holds 82 blobs. This fixture is
    therefore NOT a faithful reconstruction of the witnessed round-1 sentence
    — the witnessed claim named the NON-empty PARENT ``judge/`` (task #2191
    body + the clarifier's pre-correction record of the original line 37) and
    is pinned as a clean PASS by
    test_committed_under_silent_on_witnessed_parent_path. This test covers
    the empty-path form of the class (fabricated / wrong / never-committed
    paths), which is what the task body's falsifiability definition specifies.
    """
    line = (
        "the per-wave scores/items corpus is committed under "
        "`eval_results/issue_2162/judge/scores/` at the branch pin (`20fcef9c28…`)."
    )
    r = verify_report.check_committed_under_claims([line], _REPO_ROOT)
    assert r.passed is False
    assert "eval_results/issue_2162/judge/scores/" in r.detail
    assert "20fcef9c28" in r.detail  # names the pin
    assert "HF home" in r.detail  # the rewording guidance


@requires_2162_pin
def test_committed_under_silent_on_witnessed_parent_path():
    """Criterion 1-bis — check (a) is SILENT on the witnessed #2162 round-1
    sentence: the founding incident's witnessed shape, which this check
    DELIBERATELY does not catch.

    The witnessed round-1 claim asserted judge scores and items were committed
    under the PARENT ``eval_results/issue_2162/judge/`` — which held 82 blobs
    at ``20fcef9c28`` (zero under ``judge/scores/``) — i.e. a SUBSET claim
    over a NON-empty directory, which is invisible to a path-emptiness rule.
    Broadening the matcher was REJECTED: any mechanical subset rule (mapping
    claim nouns like "scores"/"items" to filename tokens under the path) is
    free-text semantics with a live false-FAIL channel ("gate reports
    committed under `judge/`" would FAIL though correct), forbidden by the
    task body's conservative-matcher instruction. If this test ever FAILs,
    the matcher has been broadened past its sanctioned scope — treat it as a
    REGRESSION GATE in the under-fire direction, not a TODO to fix.
    """
    line = (
        "judge scores and items were committed under "
        "`eval_results/issue_2162/judge/` at `20fcef9c28…`."
    )
    r = verify_report.check_committed_under_claims([line], _REPO_ROOT)
    assert r.passed is True
    assert r.is_warn is False


@requires_2162_pin
@requires_2162_branch
def test_committed_under_silent_on_2162_corrected():
    """Criterion 2 — check (a) is a clean PASS on the corrected #2162 lines
    (37: parent ``judge/`` at the branch pin, 82 blobs; 116: the five-member
    brace claim, 11/82/1/2/1 blobs at ``b228639eac…`` / the branch tip)."""
    r = verify_report.check_committed_under_claims([_2162_L37, _2162_L116], _REPO_ROOT)
    assert r.passed is True, r.detail
    assert r.is_warn is False, r.detail


@requires_2162_branch
def test_code_sha_cards_fires_on_2162_round1():
    """Criterion 3 — check (b) FIRES on the reconstructed round-1 defect: with
    every b4ab citation reverted to ba34, the grid/anchors phase's card commit
    (``gates/pilot_gate_report.json`` -> ``b4ab6ed5f9…``) is uncited."""
    text = _round1_reconstruction()
    r = verify_report.check_code_sha_cards(
        text,
        text.splitlines(),
        mode="generation",
        figures_root=_REPO_ROOT,
        expect_issue=2162,
    )
    assert r.passed is False
    assert "gates/pilot_gate_report.json" in r.detail
    assert "b4ab6ed5f9" in r.detail
    assert "per-phase" in r.detail and "split" in r.detail


@requires_2162_branch
def test_code_sha_cards_silent_on_2162_corrected():
    """Criterion 4 — check (b) is a clean PASS on the corrected excerpt set
    (all usable card commits cited), with the degenerate-card exclusions
    (abbreviated 8-hex judge-side records, dirty records) listed in the
    detail. Counts are deliberately NOT pinned — the card set on
    origin/issue-2162 is external mutable state."""
    text = "\n".join(_2162_CORRECTED_LINES)
    r = verify_report.check_code_sha_cards(
        text,
        text.splitlines(),
        mode="generation",
        figures_root=_REPO_ROOT,
        expect_issue=2162,
    )
    assert r.passed is True, r.detail
    assert r.is_warn is False, r.detail
    assert "usable card commit(s) all cited" in r.detail
    assert "excluded" in r.detail  # the abbreviated/dirty judge-side records


# ── Hermetic twins: check (a) ────────────────────────────────────────────────


@pytest.fixture
def claims_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A real git repo with two commits: ``sha1`` = data/ absent; ``sha2``
    (head) = data/x/f.txt + data/y/f.txt committed. Returns (root, sha1, sha2)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    _git_run(tmp_path, "config", "user.email", "test@test.test")
    _git_run(tmp_path, "config", "user.name", "Test")
    _git_run(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("base\n")
    _git_run(tmp_path, "add", "README.md")
    _git_run(tmp_path, "commit", "-q", "-m", "base")
    sha1 = _git_run(tmp_path, "rev-parse", "HEAD")
    for sub in ("x", "y"):
        d = tmp_path / "data" / sub
        d.mkdir(parents=True)
        (d / "f.txt").write_text("payload\n")
    _git_run(tmp_path, "add", "data")
    _git_run(tmp_path, "commit", "-q", "-m", "add data")
    sha2 = _git_run(tmp_path, "rev-parse", "HEAD")
    return tmp_path, sha1, sha2


def test_committed_under_empty_path_fails(claims_repo):
    root, sha1, _sha2 = claims_repo
    line = "artifacts are committed under `data/x` at `" + sha1 + "`"
    r = verify_report.check_committed_under_claims([line], root)
    assert r.passed is False
    assert sha1[:12] in r.detail
    assert "HF home" in r.detail


def test_committed_under_nonempty_path_passes(claims_repo):
    root, _sha1, sha2 = claims_repo
    line = "artifacts are committed under `data/x` at `" + sha2 + "`"
    r = verify_report.check_committed_under_claims([line], root)
    assert r.passed is True
    assert r.is_warn is False


def test_committed_under_brace_expansion(claims_repo):
    root, _sha1, sha2 = claims_repo
    ok = "committed under `data/{x, y}` at `" + sha2 + "`"
    r = verify_report.check_committed_under_claims([ok], root)
    assert r.passed is True and r.is_warn is False
    # One empty member fails the whole claim (every member must resolve).
    bad = "committed under `data/{x, z}` at `" + sha2 + "`"
    r = verify_report.check_committed_under_claims([bad], root)
    assert r.passed is False


def test_committed_under_any_pin_satisfies(claims_repo):
    """Empty at the inline SHA but non-empty at the same-line branch tip ->
    the claim PASSes (any-pin-satisfies is maximally conservative)."""
    root, sha1, _sha2 = claims_repo
    _git_run(root, "branch", "issue-5")  # tip = sha2, which has data/x
    line = "committed under `data/x` at `" + sha1 + "` on branch `issue-5`"
    r = verify_report.check_committed_under_claims([line], root)
    assert r.passed is True, r.detail
    assert r.is_warn is False


_NEGATION_VARIANTS = [
    "the corpus is not committed under `data/x` at `{sha}`",
    "nothing is committed under `data/x` at `{sha}`",
    "the corpus isn't in git under `data/x` at `{sha}`",
    "the corpus wasn't in git under `data/x` at `{sha}`",
    "these never landed in git under `data/x` at `{sha}`",
    "no longer committed under `data/x` at `{sha}`",
    "scores live on HF under `hf-repo/scores` rather than in git under `data/x` at `{sha}`",
    "scores live on HF under `hf-repo/scores` instead of in git under `data/x` at `{sha}`",
]


@pytest.mark.parametrize("template", _NEGATION_VARIANTS)
def test_negation_guard_variants_do_not_fire(claims_repo, template):
    """Each natural negation form must NOT fire — all run against an EMPTY
    path with a resolvable same-line pin, i.e. exactly the configuration that
    would FAIL without the guard. These pin _NEGATION_WINDOW_CHARS and
    _NEGATION_TOKENS as BEHAVIOR (an immediate lookbehind would be defeated
    by the intervening-word variants)."""
    root, sha1, _sha2 = claims_repo
    line = template.format(sha=sha1)
    r = verify_report.check_committed_under_claims([line], root)
    assert r.passed is True, r.detail
    assert r.is_warn is False, r.detail
    assert "negated claim skipped" in r.detail


def test_negation_guard_does_not_over_suppress(claims_repo):
    """A "not" EARLIER than the window (> _NEGATION_WINDOW_CHARS chars before
    the match) must NOT suppress: the empty-path claim still FAILs —
    otherwise the guard silently swallows genuine defects in long sentences."""
    root, sha1, _sha2 = claims_repo
    line = (
        "not one reviewer expected this layout, and moreover the files are "
        "committed under `data/x` at `" + sha1 + "`"
    )
    m = verify_report._COMMITTED_UNDER_RE.search(line)
    assert m is not None
    window = line[max(0, m.start() - verify_report._NEGATION_WINDOW_CHARS) : m.start()].lower()
    assert not any(tok in window for tok in verify_report._NEGATION_TOKENS), (
        "fixture sanity: the negation token must sit OUTSIDE the window"
    )
    r = verify_report.check_committed_under_claims([line], root)
    assert r.passed is False


def test_committed_under_url_hex_is_not_a_pin(claims_repo):
    """A 40-hex revision inside a URL span must not be mistaken for a git pin:
    here the URL names sha2 (which HAS data/x) while the inline pin is sha1
    (which lacks it) — the claim FAILs, proving the URL hex was excluded."""
    root, sha1, sha2 = claims_repo
    line = (
        "artifacts committed under `data/x` at `" + sha1 + "` "
        "(see https://example.com/tree/" + sha2 + "/data)"
    )
    r = verify_report.check_committed_under_claims([line], root)
    assert r.passed is False
    assert sha1[:12] in r.detail


def test_committed_under_no_pin_warns_with_branch_tip_probe(claims_repo):
    """No resolvable same-line pin -> WARN, and the detail carries the
    informational issue-branch-tip probe verdict (severity stays WARN — a
    FAIL here would import the deleted-later-at-tip false-FAIL class)."""
    root, _sha1, _sha2 = claims_repo
    _git_run(root, "branch", "issue-5")  # tip has data/x, lacks data/z
    present = [_detailed_link(5), "artifacts committed under `data/x` (no pin on this line)"]
    r = verify_report.check_committed_under_claims(present, root)
    assert r.passed is True and r.is_warn is True
    assert "no resolvable same-line pin" in r.detail
    assert "path resolves at `issue-5` tip" in r.detail
    absent = [_detailed_link(5), "artifacts committed under `data/z` (no pin on this line)"]
    r = verify_report.check_committed_under_claims(absent, root)
    assert r.passed is True and r.is_warn is True
    assert "path also empty at `issue-5` tip" in r.detail


def test_committed_under_non_git_root_warns(tmp_path):
    line = "artifacts committed under `data/x` at `" + "f" * 40 + "`"
    r = verify_report.check_committed_under_claims([line], tmp_path)
    assert r.passed is True and r.is_warn is True
    assert "not a git checkout" in r.detail


def test_committed_under_no_claims_is_na(tmp_path):
    """No claims -> PASS-note N/A, even on a non-git root (the claim scan
    precedes the git-checkout degrade, mirroring _check_pin_blob_identity)."""
    r = verify_report.check_committed_under_claims(["prose without the trigger"], tmp_path)
    assert r.passed is True and r.is_warn is False
    assert "N/A" in r.detail


def test_committed_under_skip_rules(claims_repo):
    """URL / absolute / ellipsis-abbreviated / slash-less paths are skipped
    with a note — never checked, never FAILed."""
    root, sha1, _sha2 = claims_repo
    lines = [
        "committed under `https://hf.co/x/y` at `" + sha1 + "`",
        "committed under `/abs/path` at `" + sha1 + "`",
        "committed under `data/…/x` at `" + sha1 + "`",
        "committed under `filename.json` at `" + sha1 + "`",
    ]
    r = verify_report.check_committed_under_claims(lines, root)
    assert r.passed is True, r.detail
    assert r.is_warn is False, r.detail
    for fragment in ("URL path", "absolute path", "abbreviated path", "slash-less path"):
        assert fragment in r.detail


# ── Hermetic twins: check (b) ────────────────────────────────────────────────

_SHA_A = "a1" * 20
_SHA_B = "b2" * 20
_SHA_C = "c3" * 20
_SHA_D = "d4" * 20


def _write_card(root: Path, rel: str, payload: dict) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


def _cards(body: str, root: Path, *, mode: str = "generation", issue: int | None = 7, lines=None):
    return verify_report.check_code_sha_cards(
        body,
        body.splitlines() if lines is None else lines,
        mode=mode,
        figures_root=root,
        expect_issue=issue,
    )


def test_card_walker_finds_all_nesting_paths_and_final_commit_sha(tmp_path):
    """The recursive key walk finds git_commit at every observed nesting depth
    (reproducibility_card / note.reproducibility_card / repro) AND the
    top-level final_commit_sha twin — never a fixed dotted path."""
    _write_card(
        tmp_path, "eval_results/issue_7/m/a.json", {"reproducibility_card": {"git_commit": _SHA_A}}
    )
    _write_card(
        tmp_path,
        "eval_results/issue_7/m/b.json",
        {"note": {"reproducibility_card": {"git_commit": _SHA_B}}},
    )
    _write_card(tmp_path, "eval_results/issue_7/m/c.json", {"repro": {"git_commit": _SHA_C}})
    _write_card(tmp_path, "eval_results/issue_7/m/d.json", {"final_commit_sha": _SHA_D})
    body = f"cites {_SHA_A} and {_SHA_B} and {_SHA_C}"  # _SHA_D uncited
    r = _cards(body, tmp_path)
    assert r.passed is False
    assert _SHA_D[:12] in r.detail
    assert "d.json" in r.detail
    r = _cards(body + f" plus {_SHA_D}", tmp_path)
    assert r.passed is True and r.is_warn is False


def test_degenerate_cards_excluded(tmp_path):
    """Dirty, "unknown", and abbreviated 8-hex card values are excluded from
    the FAIL set and listed in the PASS detail — a card writer's defective
    provenance must not punish the report."""
    _write_card(
        tmp_path,
        "eval_results/issue_7/x/dirty.json",
        {"repro": {"git_commit": _SHA_A, "git_dirty": True}},
    )
    _write_card(
        tmp_path, "eval_results/issue_7/x/unknown.json", {"repro": {"git_commit": "unknown"}}
    )
    _write_card(
        tmp_path, "eval_results/issue_7/x/abbrev.json", {"repro": {"git_commit": "abcd1234"}}
    )
    r = _cards("no hex citations at all", tmp_path)
    assert r.passed is True and r.is_warn is False, r.detail
    assert "1 dirty record(s) excluded" in r.detail
    assert "1 abbreviated (<40-hex) record(s) excluded" in r.detail
    assert "1 non-hex record(s) excluded" in r.detail


def test_b1_mode_split(tmp_path):
    """One uncited usable card: FAIL at generation, WARN (passed) at promote —
    the card set is external mutable state that may grow after authoring."""
    _write_card(tmp_path, "eval_results/issue_7/m/card.json", {"repro": {"git_commit": _SHA_A}})
    r = _cards("no citation of that commit", tmp_path, mode="generation")
    assert r.passed is False
    r = _cards("no citation of that commit", tmp_path, mode="promote")
    assert r.passed is True
    assert r.is_warn is True


def test_issue_inferred_from_detailed_writeup_line(tmp_path):
    """With no --issue/--expect-issue, the issue number comes from the
    report's own **Detailed writeup:** line — the inference that keeps the
    check live under the gate's real invocation."""
    _write_card(tmp_path, "eval_results/issue_7/m/card.json", {"repro": {"git_commit": _SHA_B}})
    lines = [_detailed_link(7), "prose that cites nothing"]
    r = _cards("\n".join(lines), tmp_path, issue=None, lines=lines)
    assert r.passed is False
    assert _SHA_B[:12] in r.detail


def test_unknown_issue_warns(tmp_path):
    r = _cards("prose with no detailed-writeup line", tmp_path, issue=None)
    assert r.passed is True and r.is_warn is True
    assert "card check skipped" in r.detail


def test_no_cards_anywhere_is_pass_note(tmp_path):
    """No eval_results/issue_<N> dir AND no resolvable issue ref -> PASS-note
    (NOT a WARN — the synthetic suite fixtures must stay warn-free)."""
    r = _cards("prose", tmp_path, issue=9)
    assert r.passed is True and r.is_warn is False
    assert "no reproducibility cards" in r.detail


def test_oversize_card_json_skipped(tmp_path):
    payload = {
        "repro": {"git_commit": _SHA_A},
        "pad": "x" * (verify_report._CARD_JSON_MAX_BYTES + 64),
    }
    _write_card(tmp_path, "eval_results/issue_7/m/big.json", payload)
    r = _cards("no citations", tmp_path)
    assert r.passed is True, r.detail  # the record inside was never read
    assert "oversize" in r.detail


def test_unparseable_card_json_skipped(tmp_path):
    (tmp_path / "eval_results" / "issue_7" / "m").mkdir(parents=True)
    (tmp_path / "eval_results" / "issue_7" / "m" / "bad.json").write_text("not json{")
    _write_card(tmp_path, "eval_results/issue_7/m/good.json", {"repro": {"git_commit": _SHA_A}})
    r = _cards(f"cites {_SHA_A}", tmp_path)
    assert r.passed is True, r.detail
    assert "unparseable" in r.detail


def test_cards_collected_from_git_ref(tmp_path):
    """The #2162 shape: cards exist ONLY at the issue-<N> ref (neither on the
    checked-out branch nor in the working tree) — S2 still finds them."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    _git_run(tmp_path, "config", "user.email", "test@test.test")
    _git_run(tmp_path, "config", "user.name", "Test")
    _git_run(tmp_path, "config", "commit.gpgsign", "false")
    _write_card(tmp_path, "eval_results/issue_7/m/card.json", {"repro": {"git_commit": _SHA_A}})
    _git_run(tmp_path, "add", "eval_results")
    _git_run(tmp_path, "commit", "-q", "-m", "card")
    _git_run(tmp_path, "branch", "issue-7")
    _git_run(tmp_path, "rm", "-r", "-q", "eval_results")
    _git_run(tmp_path, "commit", "-q", "-m", "drop from working branch")
    assert not (tmp_path / "eval_results").exists()
    r = _cards("no citations", tmp_path)
    assert r.passed is False
    assert "issue-7:eval_results/issue_7/m/card.json" in r.detail


def test_b2_row_missing_cited_sha_warns(tmp_path):
    _write_card(
        tmp_path,
        "eval_results/issue_7/margin/upload_done.json",
        {"reproducibility_card": {"git_commit": _SHA_A}},
    )
    body = f"code at `{_SHA_A}`\n| Code SHAs | `{_SHA_B}` | src |"
    r = _cards(body, tmp_path)
    assert r.passed is True and r.is_warn is True, r.detail
    assert "absent from the Code-SHAs row" in r.detail


def test_b3_mispairing_warns(tmp_path):
    """Both card SHAs cited and both in the row, but paired to the WRONG phase
    labels: the token-resolvable segments WARN with the per-phase suggestion."""
    _write_card(
        tmp_path,
        "eval_results/issue_7/margin/upload_done.json",
        {"reproducibility_card": {"git_commit": _SHA_A}},
    )
    _write_card(
        tmp_path,
        "eval_results/issue_7/stage2/stage2_results.json",
        {"repro": {"git_commit": _SHA_B}},
    )
    body = f"| Code SHAs | margin `{_SHA_B}` · stage-2 `{_SHA_A}` | src |\ncites {_SHA_A} {_SHA_B}"
    r = _cards(body, tmp_path)
    assert r.passed is True and r.is_warn is True, r.detail
    assert "resolves to card commit" in r.detail
    assert "per-phase split" in r.detail


def test_b3_stopword_set_pinned_and_enables_resolution(tmp_path):
    """The b3 stopword set is pinned VERBATIM (it is the check's only
    otherwise-ungrounded constant), and one pairing case resolves ONLY
    because a stopword was removed from the card-side token set: without
    removing "gates", the pilot card would also hit the label and the two
    hit cards would disagree -> silently skipped instead of WARNing."""
    assert (
        frozenset(
            {"report", "json", "upload", "done", "card", "sentinel", "results", "gate", "gates"}
        )
        == verify_report._CARD_TOKEN_STOPWORDS
    )
    _write_card(
        tmp_path,
        "eval_results/issue_7/margin/upload_done.json",
        {"reproducibility_card": {"git_commit": _SHA_A}},
    )
    _write_card(
        tmp_path,
        "eval_results/issue_7/gates/pilot_gate_report.json",
        {"repro": {"git_commit": _SHA_B}},
    )
    # Extra segments carry A + B so b2 (row coverage) stays silent and any
    # WARN is attributable to the b3 pairing leg alone.
    body = (
        f"| Code SHAs | margin gates `{_SHA_C}` · other `{_SHA_A}` · misc `{_SHA_B}` | src |"
        f"\ncites {_SHA_A} {_SHA_B}"
    )
    r = _cards(body, tmp_path)
    assert r.passed is True and r.is_warn is True, r.detail
    assert _SHA_A[:12] in r.detail
    assert "resolves to card commit" in r.detail
    # Counter-case: a label hitting TWO cards with DIFFERENT SHAs is
    # unresolvable and silently skipped (no WARN).
    body2 = (
        f"| Code SHAs | margin pilot `{_SHA_C}` · other `{_SHA_A}` · misc `{_SHA_B}` | src |"
        f"\ncites {_SHA_A} {_SHA_B}"
    )
    r2 = _cards(body2, tmp_path)
    assert r2.passed is True and r2.is_warn is False, r2.detail
    assert "unresolvable row segment(s) skipped" in r2.detail


# ── CLI-level (criterion 8): the real dispatch path renders both checks ─────


def test_cli_renders_new_checks_and_passes(figs_root, tmp_path):
    """Valid synthetic body through main(): rc 0, both new check names render.
    --figures-root is passed EXPLICITLY (a fixture inside the repo would
    otherwise default-resolve to the real repo root and silently depend on
    the live object DB)."""
    good = tmp_path / "good.md"
    good.write_text(_assemble(_default_sections()))
    r = subprocess.run(
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
    assert r.returncode == 0, r.stdout + r.stderr
    assert "committed-under-claims" in r.stdout
    assert "code-sha-cards" in r.stdout


def test_cli_code_sha_cards_mode_split_through_dispatch(figs_root, tmp_path):
    """The b1 severity branch through the REAL dispatch: an uncited usable
    card under the figures root FAILs the generation run (rc 1) and only
    WARNs the promote run (rc 0)."""
    _write_card(
        figs_root, "eval_results/issue_5/margin/upload_done.json", {"repro": {"git_commit": _SHA_A}}
    )
    gen = tmp_path / "gen.md"
    gen.write_text(_assemble(_default_sections()))
    base = [sys.executable, str(_SCRIPT), "--figures-root", str(figs_root)]
    r_gen = subprocess.run(
        [*base, "--file", str(gen), "--mode", "generation"], capture_output=True, text=True
    )
    assert r_gen.returncode == 1, r_gen.stdout + r_gen.stderr
    assert "[FAIL] code-sha-cards" in r_gen.stdout
    assert "OVERALL: FAIL (1 of" in r_gen.stdout  # the ONLY failing check
    prom = tmp_path / "prom.md"
    prom.write_text(_assemble(_promote_sections()))
    r_prom = subprocess.run(
        [*base, "--file", str(prom), "--mode", "promote"], capture_output=True, text=True
    )
    assert r_prom.returncode == 0, r_prom.stdout + r_prom.stderr
    assert "[WARN] code-sha-cards" in r_prom.stdout
