"""Pre-publish validator for Sagan-style HTML clean-result bodies.

The Sagan-card format is documented at ``~/sagan/docs/clean-result-guidelines.md``.
This script enforces the mechanical subset of those rules. The
clean-result-critic agent runs subjective lenses on top.

Usage
-----
    uv run python scripts/verify_sagan_card.py <path-to-body.html>
    uv run python scripts/verify_sagan_card.py --issue <N>

Exits 0 if every check is PASS or WARN; exits 1 if any FAIL.

Checks
------
1.  Scoped <style> block — body has an inline ``<style>...</style>`` with
    a ``.cr-<number>`` class namespace.
2.  TL;DR section — ``<section id="tldr">`` with ``<h2>TL;DR</h2>`` and
    exactly four top-level ``<li>`` bullets (Motivation, What I ran,
    Results, Next steps).
3.  Hero figure — ``<figure id="figure">`` with ``<svg>`` or ``<img>``
    and a non-empty ``<figcaption>``.
4.  Experimental-design dropdown — ``<details id="design">`` with
    ``<summary>Experimental design</summary>``.
5.  Reproducibility appendix — ``<details id="repro">`` present, appears
    AFTER the design dropdown, contains the three named groups
    (``Artifacts``, ``Compute``, ``Code``).
6.  URL permanence (repro block only) — every HF Hub link carries a
    ``/tree/<ref>`` or ``@<ref>``; every WandB link carries
    ``/runs/<id>``; every GitHub link with a ref carries
    ``/tree/<sha>`` or ``/blob/<sha>``.
7.  Sentinel scrub (repro block only) — no ``{{``, ``TBD``, ``see config``,
    ``default``. Use ``n/a`` for inapplicable fields.
8.  Confidence-rationale line — body contains
    ``Confidence: (LOW|MODERATE|HIGH) — <text>`` with ≥20 chars of
    rationale, BEFORE the repro block.
9.  Cherry-picked label — for every ``<pre>`` block inside ``#design``
    that holds completion-style text, the ~400 chars immediately above
    it mention ``cherry-picked for illustration`` OR explicitly disclose
    random sampling.
10. Title vs body confidence — when invoked with ``--issue N``, the
    title's ``(... confidence)`` marker must match the body's
    confidence line.
11. Qualitative-data link — for every ``<pre>`` sample block in
    ``#design``, the ~400 chars immediately above must carry at least
    one link or ``<code>``-wrapped path that does NOT match an obvious
    aggregate-only pattern (``*regression*``, ``*summary*``,
    ``*aggregat*``, ``*per[-_]cell*``, ``*.npz``). Aggregate-only
    artifacts (per-cell regression CSVs, summary JSONs) do not
    satisfy this rule — a reader auditing the cherry-picked samples
    needs access to the surrounding raw text. If raw completions
    truly cannot be uploaded, an explicit ``not uploaded`` /
    ``not available`` disclosure in the prelude downgrades FAIL to
    WARN; the next-steps bullet must mention re-running with upload.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ─── Report plumbing ─────────────────────────────────────────────────────────

Status = str  # "PASS" | "WARN" | "FAIL" | "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str


class Report:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, name: str, status: Status, detail: str = "") -> None:
        self.results.append(CheckResult(name, status, detail))

    def render(self) -> str:
        width = max(len(r.name) for r in self.results) if self.results else 30
        lines = [f"{'Check':<{width}}  Status  Detail", "-" * (width + 16 + 40)]
        for r in self.results:
            icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗", "SKIP": "·"}[r.status]
            lines.append(f"{r.name:<{width}}  {icon} {r.status:<5}  {r.detail}")
        return "\n".join(lines)

    def has_fail(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)


# ─── HTML slicing helpers (regex-only — no BS4 dep) ──────────────────────────


def slice_block(body: str, tag: str, attr_id: str) -> str | None:
    """Return the inner HTML of ``<{tag} id="{attr_id}">...</{tag}>`` or None."""
    pat = re.compile(
        rf'<{tag}\b[^>]*\bid="{re.escape(attr_id)}"[^>]*>(.*?)</{tag}>',
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(body)
    return m.group(1) if m else None


def find_block_span(body: str, tag: str, attr_id: str) -> tuple[int, int] | None:
    pat = re.compile(
        rf'<{tag}\b[^>]*\bid="{re.escape(attr_id)}"[^>]*>(.*?)</{tag}>',
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(body)
    return (m.start(), m.end()) if m else None


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s)


# ─── Checks ──────────────────────────────────────────────────────────────────


def check_style_block(body: str, report: Report) -> None:
    if not re.search(r"<style[^>]*>.*?</style>", body, re.DOTALL):
        report.add("Scoped <style> block", "FAIL", "no inline <style> block found")
        return
    if not re.search(r"\.cr-\d+\b", body):
        report.add(
            "Scoped <style> block",
            "WARN",
            "no .cr-<number> namespace — CSS may leak into the dashboard chrome",
        )
        return
    report.add("Scoped <style> block", "PASS", "")


def check_tldr_section(body: str, report: Report) -> None:
    inner = slice_block(body, "section", "tldr")
    if inner is None:
        report.add("TL;DR section", "FAIL", '<section id="tldr"> missing')
        return
    if not re.search(r"<h2[^>]*>\s*TL;DR\s*</h2>", inner, re.IGNORECASE):
        report.add("TL;DR section", "FAIL", "<h2>TL;DR</h2> missing inside #tldr")
        return
    # Top-level <li>: count <li> directly under the outer <ul>, ignoring
    # nested <ul> children (Next-steps sub-bullets are allowed). Greedy
    # outer match so we capture through the OUTER </ul>; then iteratively
    # strip innermost nested <ul>...</ul> until only top-level li remain.
    ul_m = re.search(r"<ul[^>]*>(.*)</ul>", inner, re.DOTALL)
    if ul_m is None:
        report.add("TL;DR section", "FAIL", "no <ul> inside #tldr")
        return
    ul_body = ul_m.group(1)
    while True:
        new = re.sub(r"<ul[^>]*>[^<]*(?:<(?!/?ul\b)[^<]*)*</ul>", "", ul_body, flags=re.DOTALL)
        if new == ul_body:
            break
        ul_body = new
    top_lis = re.findall(r"<li\b", ul_body)
    if len(top_lis) != 4:
        report.add(
            "TL;DR section",
            "FAIL",
            f"expected 4 top-level bullets (Motivation / What I ran / Results / Next steps), found {len(top_lis)}",
        )
        return
    expected_labels = ["Motivation", "What I ran", "Results", "Next steps"]
    missing = [lbl for lbl in expected_labels if lbl not in strip_tags(inner)]
    if missing:
        report.add(
            "TL;DR section",
            "WARN",
            f"expected bullet labels not found verbatim: {', '.join(missing)}",
        )
        return
    report.add("TL;DR section", "PASS", "4 bullets with expected labels")


def check_hero_figure(body: str, report: Report) -> None:
    inner = slice_block(body, "figure", "figure")
    if inner is None:
        report.add("Hero figure", "FAIL", '<figure id="figure"> missing')
        return
    has_visual = bool(re.search(r"<(svg|img)\b", inner, re.IGNORECASE))
    if not has_visual:
        report.add("Hero figure", "FAIL", "no <svg> or <img> inside #figure")
        return
    cap_m = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", inner, re.DOTALL)
    if cap_m is None:
        report.add("Hero figure", "FAIL", "no <figcaption> inside #figure")
        return
    cap_text = strip_tags(cap_m.group(1)).strip()
    if len(cap_text.split()) < 10:
        report.add(
            "Hero figure",
            "FAIL",
            f"figcaption is too short ({len(cap_text.split())} words; need ≥10)",
        )
        return
    report.add("Hero figure", "PASS", f"figcaption has {len(cap_text.split())} words")


def check_design_block(body: str, report: Report) -> None:
    inner = slice_block(body, "details", "design")
    if inner is None:
        report.add("Experimental design block", "FAIL", '<details id="design"> missing')
        return
    if not re.search(r"<summary[^>]*>\s*Experimental design\s*</summary>", inner, re.IGNORECASE):
        report.add(
            "Experimental design block",
            "WARN",
            '<summary> text is not "Experimental design"',
        )
        return
    report.add("Experimental design block", "PASS", "")


def check_repro_block(body: str, report: Report) -> tuple[str | None, tuple[int, int] | None]:
    inner = slice_block(body, "details", "repro")
    span = find_block_span(body, "details", "repro")
    if inner is None:
        report.add(
            "Reproducibility appendix",
            "FAIL",
            '<details id="repro"> missing — required at the bottom of the body',
        )
        return None, None
    if not re.search(
        r"<summary[^>]*>\s*Reproducibility\b.*?</summary>", inner, re.IGNORECASE | re.DOTALL
    ):
        report.add(
            "Reproducibility appendix",
            "WARN",
            '<summary> text should start with "Reproducibility"',
        )
    # Three required groups
    text = strip_tags(inner)
    missing = [g for g in ("Artifacts", "Compute", "Code") if g not in text]
    if missing:
        report.add(
            "Reproducibility appendix",
            "FAIL",
            f"missing required groups: {', '.join(missing)}",
        )
        return inner, span
    # Position relative to design block
    design_span = find_block_span(body, "details", "design")
    if design_span and span and span[0] < design_span[1]:
        report.add(
            "Reproducibility appendix",
            "FAIL",
            "repro block appears before/inside the design block — must be AFTER",
        )
        return inner, span
    report.add(
        "Reproducibility appendix",
        "PASS",
        "Artifacts + Compute + Code present, positioned after #design",
    )
    return inner, span


def check_url_permanence(repro_inner: str | None, report: Report) -> None:
    if repro_inner is None:
        report.add("URL permanence", "SKIP", "no repro block to inspect")
        return
    issues: list[str] = []
    for url in re.findall(r'href="([^"]+)"', repro_inner):
        if "huggingface.co" in url:
            if not re.search(r"/tree/[^/?#]+|@[^/?#]+", url) and not re.search(r"/blob/[^/]+", url):
                issues.append(f"HF Hub URL not pinned to a ref: {url}")
        elif "wandb.ai" in url:
            if not re.search(r"/runs/[^/?#]+", url):
                issues.append(f"WandB URL not pointing at a specific run: {url}")
        elif "github.com" in url:
            if re.search(r"/blob/(main|master)\b", url) or re.search(r"/tree/(main|master)\b", url):
                issues.append(f"GitHub URL pinned to a moving branch: {url}")
    if issues:
        report.add(
            "URL permanence", "FAIL", "; ".join(issues[:3]) + (" …" if len(issues) > 3 else "")
        )
        return
    report.add("URL permanence", "PASS", "all repro URLs pin to permanent refs")


SENTINELS = ("{{", "TBD", "see config", "default")


def check_sentinel_scrub(repro_inner: str | None, report: Report) -> None:
    if repro_inner is None:
        report.add("Sentinel scrub", "SKIP", "no repro block to inspect")
        return
    text = strip_tags(repro_inner)
    found = [s for s in SENTINELS if s.lower() in text.lower()]
    # "default" appears in plenty of legitimate code; require it as a standalone token
    if "default" in found and not re.search(r"\bdefault\b", text):
        found.remove("default")
    if found:
        report.add(
            "Sentinel scrub",
            "FAIL",
            f"placeholder tokens in repro block: {', '.join(found)} — use n/a explicitly",
        )
        return
    report.add("Sentinel scrub", "PASS", "no placeholder tokens")


CONFIDENCE_LINE = re.compile(
    r"Confidence:\s*(LOW|MODERATE|HIGH)\s*[—–\-]\s*(.{20,})",
    re.IGNORECASE,
)


def check_confidence_line(
    body: str, repro_span: tuple[int, int] | None, report: Report
) -> str | None:
    haystack = body[: repro_span[0]] if repro_span else body
    m = CONFIDENCE_LINE.search(strip_tags(haystack))
    if m is None:
        report.add(
            "Confidence rationale line",
            "FAIL",
            'no "Confidence: LOW|MODERATE|HIGH — <rationale>" sentence before #repro',
        )
        return None
    report.add("Confidence rationale line", "PASS", f"label={m.group(1).upper()}")
    return m.group(1).upper()


def _sample_prelude(design_inner: str, pre_start: int, max_window: int = 1500) -> str:
    """Return the prelude HTML for a <pre> sample block.

    The window covers the enclosing <p>/<li> (if any open within ``max_window``
    chars before the <pre>) plus the immediately preceding paragraph. Falls
    back to a ``max_window``-char window when no opening tag is found —
    handles prose paragraphs that exceed the 400-char default.
    """
    lo = max(0, pre_start - max_window)
    chunk = design_inner[lo:pre_start]
    # Find the LAST opening <p> or <li> in the chunk — that's the start of
    # the immediately-preceding paragraph.
    opens = list(re.finditer(r"<(p|li|div)\b[^>]*>", chunk, re.IGNORECASE))
    if opens:
        return chunk[opens[-1].start() :]
    return chunk


def _is_sample_pre(content: str) -> bool:
    """Heuristic: completion-style if contains a User/Assistant marker or is
    long enough (>200 stripped chars). Otherwise probably a code/CLI snippet."""
    return (
        bool(re.search(r"\b(User|Assistant|Human|Model):", content, re.IGNORECASE))
        or len(strip_tags(content).strip()) > 200
    )


def check_cherry_picked_label(body: str, report: Report) -> None:
    design_inner = slice_block(body, "details", "design")
    if design_inner is None:
        report.add("Cherry-picked label", "SKIP", "no design block")
        return
    pre_blocks = list(re.finditer(r"<pre\b[^>]*>(.*?)</pre>", design_inner, re.DOTALL))
    if not pre_blocks:
        report.add("Cherry-picked label", "SKIP", "no <pre> sample blocks in design")
        return
    flagged: list[str] = []
    for m in pre_blocks:
        content = m.group(1)
        if not _is_sample_pre(content):
            continue
        prelude = strip_tags(_sample_prelude(design_inner, m.start())).lower()
        if "cherry-picked" in prelude or "cherry picked" in prelude:
            continue
        if re.search(r"\b(random[-\s]?sample|first \d+ of|drawn at random|random draw)", prelude):
            continue
        flagged.append(content.strip().splitlines()[0][:60])
    if flagged:
        report.add(
            "Cherry-picked label",
            "FAIL",
            f"{len(flagged)} sample block(s) lack 'cherry-picked for illustration' or random-sample disclosure",
        )
        return
    report.add("Cherry-picked label", "PASS", f"{len(pre_blocks)} <pre> blocks labelled or n/a")


_AGGREGATE_PATH_RE = re.compile(
    # Filenames whose stem advertises aggregation, OR the .npz extension.
    r"\b\S*(?:regression|summary|aggregat\w*|per[-_]?cell|cell[-_]?level)\S*\.(?:csv|json|jsonl|tsv|parquet|npz)\b"
    r"|\b\S+\.npz\b",
    re.IGNORECASE,
)

_NOT_UPLOADED_RE = re.compile(
    r"(?:not\s+uploaded|not\s+available|did\s+not\s+upload"
    r"|raw\s+completions?\s+(?:were\s+)?(?:not|never)"
    r"|raw[-_\s]?completions?\s+(?:were\s+)?n/a)",
    re.IGNORECASE,
)

_PRELUDE_TOKEN_RE = re.compile(
    r'href="([^"]+)"|<code[^>]*>([^<]+)</code>',
    re.IGNORECASE,
)


def check_qualitative_data_link(body: str, report: Report) -> None:
    """Check 11: every <pre> sample block in #design must be preceded by a link
    to the full qualitative-data artifact. Aggregate-only paths fail; an
    explicit 'not uploaded' disclosure downgrades the failure to WARN.
    """
    design_inner = slice_block(body, "details", "design")
    if design_inner is None:
        report.add("Qualitative-data link", "SKIP", "no design block")
        return
    pre_blocks = list(re.finditer(r"<pre\b[^>]*>(.*?)</pre>", design_inner, re.DOTALL))
    if not pre_blocks:
        report.add("Qualitative-data link", "SKIP", "no <pre> sample blocks in design")
        return

    fails: list[str] = []
    warns: list[str] = []
    passes = 0
    for m in pre_blocks:
        content = m.group(1)
        if not _is_sample_pre(content):
            continue
        prelude_raw = _sample_prelude(design_inner, m.start())
        prelude_text = strip_tags(prelude_raw)
        first_line = content.strip().splitlines()[0][:60]

        tokens = [href or code for href, code in _PRELUDE_TOKEN_RE.findall(prelude_raw)]
        has_escape = bool(_NOT_UPLOADED_RE.search(prelude_text))

        if not tokens:
            if has_escape:
                warns.append(f"'{first_line}': no link, 'not uploaded' escape acknowledged")
            else:
                fails.append(f"'{first_line}': no link or path in prelude paragraph")
            continue

        qualitative_hit = any(not _AGGREGATE_PATH_RE.search(tok) for tok in tokens)
        if qualitative_hit:
            passes += 1
            continue

        # All tokens match the aggregate-only pattern.
        if has_escape:
            warns.append(
                f"'{first_line}': only aggregate-pattern links, 'not uploaded' escape acknowledged"
            )
        else:
            fails.append(
                f"'{first_line}': only aggregate-pattern links (e.g. {tokens[0][:60]}); "
                "raw text-level artifact required"
            )

    if fails:
        report.add(
            "Qualitative-data link",
            "FAIL",
            f"{len(fails)} sample block(s) lack a qualitative-data link: "
            + "; ".join(fails[:2])
            + (" …" if len(fails) > 2 else ""),
        )
        return
    if warns:
        report.add(
            "Qualitative-data link",
            "WARN",
            f"{len(warns)} sample block(s) ship with 'not uploaded' escape — follow-up should re-run with raw-completion upload",
        )
        return
    report.add(
        "Qualitative-data link",
        "PASS",
        f"{passes} sample block(s) link to a qualitative-data artifact",
    )


def check_title_confidence(title: str | None, body_confidence: str | None, report: Report) -> None:
    if title is None:
        report.add("Title confidence match", "SKIP", "title not provided")
        return
    m = re.search(r"\((LOW|MODERATE|HIGH)\s+confidence\)\s*$", title, re.IGNORECASE)
    if m is None:
        report.add(
            "Title confidence match",
            "FAIL",
            'title does not end with "(LOW|MODERATE|HIGH confidence)"',
        )
        return
    title_label = m.group(1).upper()
    if body_confidence is None:
        report.add(
            "Title confidence match", "SKIP", "body confidence-line check failed; cannot compare"
        )
        return
    if title_label != body_confidence:
        report.add(
            "Title confidence match",
            "FAIL",
            f"title={title_label} but body Confidence:={body_confidence}",
        )
        return
    report.add("Title confidence match", "PASS", f"both = {title_label}")


# ─── Entrypoint ──────────────────────────────────────────────────────────────


def load_body_and_title(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.issue is not None:
        # Importable as a sibling module when run from the repo root via `uv run`.
        sys.path.insert(0, str(Path(__file__).parent))
        from sagan_state import get_experiment

        exp = get_experiment(args.issue)["experiment"]
        return exp.get("body") or "", exp.get("title")
    body = Path(args.path).read_text()
    return body, args.title


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("path", nargs="?", help="path to a .html body file")
    src.add_argument("--issue", type=int, help="fetch body from Sagan experiment number")
    p.add_argument("--title", help="experiment title (only when reading from a file)")
    p.add_argument("--json", action="store_true", help="emit JSON report instead of text")
    args = p.parse_args()

    body, title = load_body_and_title(args)
    report = Report()

    check_style_block(body, report)
    check_tldr_section(body, report)
    check_hero_figure(body, report)
    check_design_block(body, report)
    repro_inner, repro_span = check_repro_block(body, report)
    check_url_permanence(repro_inner, report)
    check_sentinel_scrub(repro_inner, report)
    body_conf = check_confidence_line(body, repro_span, report)
    check_cherry_picked_label(body, report)
    check_qualitative_data_link(body, report)
    check_title_confidence(title, body_conf, report)

    if args.json:
        json.dump(
            {
                "results": [
                    {"name": r.name, "status": r.status, "detail": r.detail} for r in report.results
                ],
                "fail": report.has_fail(),
            },
            sys.stdout,
            indent=2,
        )
        print()
    else:
        print(report.render())

    return 1 if report.has_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
