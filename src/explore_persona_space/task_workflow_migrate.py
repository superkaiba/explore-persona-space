"""task_workflow_migrate — migration helpers for `task.py migrate-body`.

Patches awaiting_promotion bodies into compliance with the 13-check
`verify_task_body.py` markdown spec. Two patch modes:

(a) Conformant-but-failing remediation — applied to bodies that already
    carry the four required H2 sections in order (TL;DR / Figure /
    Details / Reproducibility) but fail one or more of the content-level
    checks (Repro subgroups missing, cherry-picked label missing on a
    sample-output fence, qualitative-data link missing on a sample-output
    fence).

(b) v4-legacy shape conversion — for bodies still on the pre-2026-05-13
    `## TL;DR / ## Summary / ## Details / ## Source issues` shape.
    Snapshots the original body, then:
      - strips any `<details open><summary>...</summary>` wrappers around
        each H2 (decorative v4-era toggles),
      - decides where `## Summary` goes: if it contains a Markdown image,
        split into a leading `## Figure` (image + caption) + remainder
        folded into `## Details`; otherwise fold the whole Summary into
        `## Details` and inject a stub `## Figure` with `n/a` placeholder
        text + a generic caption,
      - injects a `## Reproducibility` section with `n/a` Artifacts /
        Compute / Code subgroups if one is not already present.

Idempotency: every transformation is a string operation guarded by a
"would this change anything?" check; running `--apply` on an already-PASS
body is a no-op (no git diff).

The module exposes one entry point — `migrate_one(task_id, *, apply,
shape=None)` — which loads the task body, classifies it, runs the
appropriate patch chain, optionally writes via `task_workflow.set_body`
(with `snapshot_original=True` on the first apply for v4-legacy shape
conversions), and returns a `MigrateResult` summary suitable for
`--report` rendering.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml as _yaml

# Make sibling scripts/ importable so we can call into verify_task_body.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import verify_task_body as vtb  # noqa: E402

from explore_persona_space import task_workflow as tw  # noqa: E402

# ─── Classification ───────────────────────────────────────────────────────


class BodyClass(Enum):
    """Body shape classification — drives which patch chain (if any) runs."""

    PASS = "pass"  # already passes verify_task_body
    LEGACY_HTML = "legacy-html"  # carries <!-- legacy-sagan-card --> sentinel
    CONFORMANT_FAILING = "conformant-failing"  # four-H2 shape, but FAILs ≥1 check
    V4_LEGACY = "v4-legacy"  # ## TL;DR / ## Summary / ## Details / ## Source issues
    UNKNOWN = "unknown"  # neither of the above


V4_LEGACY_H2 = ("TL;DR", "Summary", "Details", "Source issues")
TARGET_H2 = ("TL;DR", "Figure", "Details", "Reproducibility")


# ─── Reporting / result type ──────────────────────────────────────────────


@dataclass
class MigrateResult:
    task_id: int
    classification: BodyClass
    verify_before: str  # "PASS" / "FAIL" / "SKIP"
    verify_after: str  # "PASS" / "FAIL" / "SKIP" / "DRY-RUN"
    actions: list[str] = field(default_factory=list)
    needs_user: bool = False
    needs_user_reason: str = ""
    diff_preview: str = ""

    def report_line(self) -> str:
        tag = self.classification.value
        flag = " [needs-user]" if self.needs_user else ""
        return (
            f"#{self.task_id:<5} | {tag:<20} | before={self.verify_before:<5} "
            f"after={self.verify_after:<7}{flag}"
        )


# ─── Pre-classification helpers ───────────────────────────────────────────


def _h2_names_in_order(body: str) -> list[str]:
    """List the H2 section names in `body` in document order (no filtering)."""
    return [name for name, _, _ in vtb.find_h2_sections(body)]


def _has_legacy_sentinel(body: str) -> bool:
    return vtb.LEGACY_SAGAN_CARD_SENTINEL in body


def _is_v4_legacy(body: str) -> bool:
    """The v4-legacy shape: H2s in the order `TL;DR / Summary / Details / Source issues`."""
    h2s = _h2_names_in_order(body)
    # Allow extra H2s at the end (some bodies append more sections).
    if len(h2s) < 4:
        return False
    head = h2s[:4]
    return tuple(head) == V4_LEGACY_H2


def _is_target_shape(body: str) -> bool:
    """The target shape: at least the four required H2s in the right order."""
    h2s = _h2_names_in_order(body)
    seq = [s for s in h2s if s in TARGET_H2]
    return seq == list(TARGET_H2)


def classify_body(body: str, fm: dict | None = None) -> BodyClass:
    """Return the migration classification for `body` (a raw post-frontmatter string).

    `fm` is the task's actual frontmatter dict. Passed verbatim to the
    verifier so soft checks that key off frontmatter (e.g. the
    Goal-of-experiment soft INFO check) read the real values rather
    than a synthesized empty mapping. Callers that don't have a real
    frontmatter handy (direct fixture calls in tests, exploratory CLI
    use) can omit it; the synthesized empty mapping is fine because
    Goal presence is a WARN, not a FAIL.
    """
    if _has_legacy_sentinel(body):
        return BodyClass.LEGACY_HTML
    fm_text = _serialize_frontmatter(fm)
    overall, _ = vtb.verify_text(fm_text + body)
    if overall:
        return BodyClass.PASS
    if _is_v4_legacy(body):
        return BodyClass.V4_LEGACY
    if _is_target_shape(body):
        return BodyClass.CONFORMANT_FAILING
    return BodyClass.UNKNOWN


def _serialize_frontmatter(fm: dict | None) -> str:
    """Render `fm` back into a `---\\n...\\n---\\n` block for `verify_text`.

    Empty dict / None → `---\\n---\\n` (the historical synthesized form).
    """
    if not fm:
        return "---\n---\n"
    payload = _yaml.safe_dump(fm, sort_keys=False).strip()
    return f"---\n{payload}\n---\n"


# ─── Conformant-but-failing remediation ───────────────────────────────────


_REPRO_HEADING_RE = re.compile(r"^## Reproducibility\s*$", re.MULTILINE)


def _find_section_span(body: str, heading: str) -> tuple[int, int] | None:
    """Locate the byte span of an H2 section's body content (between the H2 line
    and the next H2 or end-of-body). Returns None if the section is missing.
    """
    target = f"## {heading}"
    lines = body.splitlines(keepends=True)
    in_target = False
    start = -1
    end = len(body)
    cursor = 0
    for line in lines:
        stripped = line.strip()
        if not in_target and stripped == target:
            in_target = True
            start = cursor + len(line)  # skip past the heading line itself
        elif in_target and stripped.startswith("## ") and not stripped.startswith("### "):
            end = cursor
            break
        cursor += len(line)
    if not in_target:
        return None
    return start, end


def remediate_repro_subgroups(body: str) -> tuple[str, list[str]]:
    """Inject missing **Artifacts:** / **Compute:** / **Code:** bold labels.

    Heuristics handled:
      (a) section uses `### Artifacts` / `### Compute` / `### Code` H3 headings
          → rewrite the heading line to a `**Label:**` bold label.
      (b) section uses `**Artifacts.**` (period instead of colon)
          → rewrite to `**Artifacts:**`.
      (c) subgroup label absent entirely
          → append `**<Label>:** n/a` as a standalone line at the end of the
            section.

    Idempotent — if the section already has all three bold labels in the
    accepted shape, returns the body unchanged.
    """
    actions: list[str] = []
    span = _find_section_span(body, "Reproducibility")
    if span is None:
        return body, actions
    start, end = span
    repro = body[start:end]

    new_repro = repro

    for label in vtb.REPRO_SUBGROUPS:
        # Already a properly-formed bold label?
        if re.search(rf"\*\*\s*{re.escape(label)}\s*:?\s*\*\*", new_repro):
            continue
        # Try heuristic (a): H3 heading promoted to bold label.
        h3_re = re.compile(rf"^### {re.escape(label)}\s*$", re.MULTILINE)
        if h3_re.search(new_repro):
            new_repro = h3_re.sub(f"**{label}:**", new_repro)
            actions.append(f"promote `### {label}` → `**{label}:**`")
            continue
        # Try heuristic (b): `**Artifacts.**` (period not colon).
        dot_re = re.compile(rf"\*\*\s*{re.escape(label)}\s*\.\s*\*\*")
        if dot_re.search(new_repro):
            new_repro = dot_re.sub(f"**{label}:**", new_repro)
            actions.append(f"fix punctuation `**{label}.**` → `**{label}:**`")
            continue
        # Heuristic (c): append a stub at the end of the section.
        stub = f"\n**{label}:** n/a\n"
        new_repro = new_repro.rstrip("\n") + stub
        actions.append(f"inject missing `**{label}:** n/a`")

    if new_repro == repro:
        return body, actions

    return body[:start] + new_repro + body[end:], actions


_RAW_COMPLETIONS_LINK_RE = re.compile(
    # Anchor on the words "raw completion(s)" then a URL on the same bullet /
    # paragraph. Group 1 = the http(s) URL (read to whitespace, `)`, `]`,
    # whitespace, or end-of-line — markdown link-syntax `(https://...)` works
    # because we stop at the closing paren).
    r"raw[-_ ]?completions?.{0,400}?(https?://[^\s\)\]\>]+)",
    re.IGNORECASE | re.DOTALL,
)


def _find_raw_completions_url(body: str) -> str | None:
    """Look in `## Reproducibility` for a URL flagged as raw-completions.

    Returns the first such URL, or None.
    """
    span = _find_section_span(body, "Reproducibility")
    if span is None:
        return None
    repro = body[span[0] : span[1]]
    m = _RAW_COMPLETIONS_LINK_RE.search(repro)
    if not m:
        return None
    return m.group(1)


def _disclosure_paragraph(body: str) -> str:
    """Choose the right qual-data disclosure for this body.

    If we can find a `raw completions: <URL>` line in Reproducibility, link
    to it inline (preserves factuality). Otherwise fall back to a
    `Raw completions not uploaded` escape (downgrades verifier FAIL → WARN).
    """
    url = _find_raw_completions_url(body)
    if url:
        return f"Raw completions are available at [{url}]({url}).\n"
    return "Raw completions not uploaded for this experiment (see Next-steps in the TL;DR).\n"


def remediate_qual_data_link(body: str) -> tuple[str, list[str]]:
    """For every sample-output fenced block in `## Details` whose prelude is
    missing a qualitative-data link (per `verify_task_body.check_qualitative_data_link`),
    insert a `Raw completions not uploaded` disclosure paragraph immediately
    before the fence and add a `- Re-run with raw-completion upload` bullet
    to the TL;DR Next-steps bullet (idempotent).

    The disclosure downgrades the verifier verdict from FAIL → WARN.
    """
    actions: list[str] = []
    details_span = _find_section_span(body, "Details")
    if details_span is None:
        return body, actions
    d_start, d_end = details_span
    details = body[d_start:d_end]

    samples = vtb._iter_sample_fences(details)
    if not samples:
        return body, actions

    disclosure = _disclosure_paragraph(body)
    used_uploaded_url = disclosure.startswith("Raw completions are available at")

    # Walk fences right-to-left so we don't invalidate earlier offsets.
    new_details = details
    inserts = 0
    for fence_start, _fence_end, _content in reversed(samples):
        prelude = vtb._prelude_window(new_details, fence_start)
        # Already has a link/path token that isn't aggregate-only?
        link_tokens = vtb._LINK_RE.findall(prelude) + vtb._CODE_RE.findall(prelude)
        qual_hit = any(not vtb._AGGREGATE_PATH_RE.search(t) for t in link_tokens)
        if qual_hit:
            continue
        # Already has a `not uploaded`-style escape?
        if vtb._NOT_UPLOADED_RE.search(prelude):
            continue
        # Walk back to find the start of the line containing fence_start.
        line_start = new_details.rfind("\n", 0, fence_start) + 1
        insert_at = line_start
        new_details = new_details[:insert_at] + disclosure + "\n" + new_details[insert_at:]
        inserts += 1

    if inserts == 0:
        return body, actions

    if used_uploaded_url:
        actions.append(f"inject raw-completions link disclosure above {inserts} sample block(s)")
    else:
        actions.append(
            f"inject `Raw completions not uploaded` disclosure above {inserts} sample block(s)"
        )
    new_body = body[:d_start] + new_details + body[d_end:]

    # Only add the Next-steps "re-run with raw-completion upload" bullet when
    # we couldn't find an uploaded URL — otherwise we'd suggest a re-run that
    # doesn't apply.
    if not used_uploaded_url:
        new_body, tldr_actions = _append_tldr_next_steps_bullet(
            new_body, "Re-run with raw-completion upload"
        )
        actions.extend(tldr_actions)
    return new_body, actions


def _append_tldr_next_steps_bullet(body: str, text: str) -> tuple[str, list[str]]:
    """Append `- Next steps: ... and <text>` to the TL;DR's Next-steps bullet,
    or insert a new `- Next steps: <text>` bullet if none exists.

    Idempotent — if the bullet already mentions `<text>`, returns body unchanged.
    """
    span = _find_section_span(body, "TL;DR")
    if span is None:
        return body, []
    start, end = span
    tldr = body[start:end]
    if text.lower() in tldr.lower():
        return body, []
    # Locate the existing "Next steps:" bullet, if any.
    next_steps_re = re.compile(
        r"(?im)^(\s*[-*]\s*(?:\*\*)?Next steps(?:\*\*)?\s*:)(.*?)(\n(?:\s*[-*]|\s*\n|##|$))",
        re.DOTALL,
    )
    m = next_steps_re.search(tldr)
    if m:
        # Append to the existing bullet.
        head, current, tail = m.group(1), m.group(2), m.group(3)
        # Strip trailing punctuation/whitespace, append ", <text>."
        current = current.rstrip(" .;\n")
        if current and not current.endswith(("—", "-", ":")):
            new_clause = f"{current}; {text}."
        else:
            new_clause = f"{current} {text}." if current else f" {text}."
        new_tldr = tldr[: m.start()] + head + new_clause + tail + tldr[m.end() :]
        new_body = body[:start] + new_tldr + body[end:]
        return new_body, [f"append `{text}` to TL;DR Next-steps bullet"]
    # No Next-steps bullet — append one at the end of the TL;DR section body.
    new_tldr = tldr.rstrip("\n") + f"\n- Next steps: {text}.\n\n"
    new_body = body[:start] + new_tldr + body[end:]
    return new_body, [f"insert `Next steps: {text}` bullet into TL;DR"]


# ─── v4-legacy → target-shape conversion ──────────────────────────────────


_DETAILS_WRAPPER_RE = re.compile(
    r"<details\s+open>\s*\n<summary>\s*\n\s*\n(##[^\n]+)\n\s*\n</summary>",
    re.MULTILINE,
)
_DETAILS_CLOSE_RE = re.compile(r"^\s*</details>\s*\n", re.MULTILINE)


def strip_v4_details_wrappers(body: str) -> tuple[str, list[str]]:
    """Strip `<details open><summary>## H2</summary>` / `</details>` toggles.

    Replaces each `<details open>...<summary>## TL;DR</summary>` block with
    a bare `## TL;DR` line; removes matching `</details>` closers.
    Idempotent — re-running on stripped output yields no further change.
    """
    actions: list[str] = []
    out = _DETAILS_WRAPPER_RE.sub(lambda m: m.group(1), body)
    if out != body:
        n = len(_DETAILS_WRAPPER_RE.findall(body))
        actions.append(f"strip {n} `<details open><summary>## H2</summary>` toggle wrapper(s)")
    body = out
    # Strip `</details>` closers (we may leave behind a few that were paired
    # with the toggles we just rewrote). NOT all `</details>` closers — only
    # the bare ones on their own line. A `<details>` block we did NOT rewrite
    # would still close itself fine if its opening tag remains; we walk and
    # only strip closers whose matching opener was rewritten.
    # Conservative approach: count opens vs closes; only strip extras.
    opens = body.count("<details")
    closes = body.count("</details>")
    if closes > opens:
        extras = closes - opens
        # Strip the first `extras` standalone `</details>` lines.
        new_body = body
        for _ in range(extras):
            new_body, n = _DETAILS_CLOSE_RE.subn("", new_body, count=1)
            if n == 0:
                break
        if new_body != body:
            actions.append(f"strip {extras} now-orphaned `</details>` closer line(s)")
        body = new_body
    return body, actions


def convert_v4_to_target(body: str, *, title: str | None = None) -> tuple[str, list[str]]:
    """Run the full v4-legacy → target-shape conversion on `body`.

    Steps:
      1. Strip `<details open><summary>## H2</summary>` toggles.
      2. Inject an H1 from the frontmatter `title` if the body has none.
      3. Rewrite `## Summary` to either `## Figure` + folded prose into
         `## Details`, depending on whether the section contains an image.
         If no image, inject a stub `## Figure` with `n/a` text.
      4. Inject `## Reproducibility` with `n/a` subgroups if absent.
    """
    actions: list[str] = []
    body, strip_actions = strip_v4_details_wrappers(body)
    actions.extend(strip_actions)

    body, h1_actions = _ensure_h1(body, title)
    actions.extend(h1_actions)

    body, summary_actions = _convert_summary_section(body)
    actions.extend(summary_actions)

    body, repro_actions = _ensure_repro_section(body)
    actions.extend(repro_actions)
    return body, actions


def _ensure_h1(body: str, title: str | None) -> tuple[str, list[str]]:
    """Inject `# <title>` as the first non-blank line of `body` if the body has
    no H1 already.
    """
    actions: list[str] = []
    if vtb.find_h1_title(body):
        return body, actions
    if not title:
        return body, actions
    # Normalize title to a single line (YAML folded-block style produces \n).
    one_line = " ".join(title.split())
    new_body = f"# {one_line}\n\n" + body.lstrip("\n")
    actions.append("inject `# <title>` H1 from frontmatter title")
    return new_body, actions


def _convert_summary_section(body: str) -> tuple[str, list[str]]:
    """Either rename `## Summary` → `## Figure` (if it contains an image), or
    fold Summary content into Details + inject a stub `## Figure`.
    """
    actions: list[str] = []
    summary_span = _find_section_span(body, "Summary")
    if summary_span is None:
        # No `## Summary` to convert. If the body has no `## Figure`, inject a stub.
        if _find_section_span(body, "Figure") is None:
            # Inject right before `## Details` if present, else before any extra H2.
            body, inj = _inject_stub_figure(body)
            actions.extend(inj)
        return body, actions

    s_start, s_end = summary_span
    summary_text = body[s_start:s_end].strip()

    has_image = bool(vtb._IMAGE_RE.search(summary_text))

    if has_image:
        # Split: keep the image + caption as `## Figure`, fold prose into `## Details`.
        image_match = vtb._IMAGE_RE.search(summary_text)
        assert image_match is not None  # narrowed by has_image
        img_start = image_match.start()
        # Find a caption line after the image
        post_image = summary_text[image_match.end() :]
        # Use the first paragraph after the image as the caption
        caption = post_image.lstrip("\n").split("\n\n", 1)[0].strip() or "Figure caption."
        # Find non-image prose before & after
        prefix_prose = summary_text[:img_start].strip()
        # Anything past the first paragraph after the image is folded into Details.
        rest_after_caption = ""
        if "\n\n" in post_image.lstrip("\n"):
            rest_after_caption = post_image.lstrip("\n").split("\n\n", 1)[1].strip()
        figure_block = f"\n\n{image_match.group(0)}\n\n*{caption}*\n\n"
        # Build new body: replace the `## Summary` heading + content with `## Figure` + content.
        # Heading line itself is the previous line at the start position.
        body, replaced = _replace_h2_section(body, "Summary", "Figure", figure_block)
        if replaced:
            actions.append("rename `## Summary` → `## Figure` (image preserved)")
        # Fold prefix_prose + rest_after_caption into Details.
        folded = "\n\n".join(p for p in (prefix_prose, rest_after_caption) if p)
        if folded:
            body, fa = _append_to_details(body, folded)
            actions.extend(fa)
    else:
        # Prose-only Summary: fold whole thing into Details, inject stub Figure.
        body, fa = _append_to_details(body, summary_text)
        actions.extend(fa)
        body, replaced = _replace_h2_section(
            body,
            "Summary",
            "Figure",
            (
                "\n\n*Figure not applicable for this experiment "
                "(no headline plot was produced).*\n\n"
                "n/a\n\n"
            ),
        )
        if replaced:
            actions.append(
                "fold prose `## Summary` into `## Details`; inject `## Figure` stub "
                "(needs human follow-up for hero image)"
            )
    return body, actions


def _replace_h2_section(
    body: str, old_name: str, new_name: str, new_section_content: str
) -> tuple[str, bool]:
    """Rename an `## old_name` heading to `## new_name` and replace its content.

    Returns (new_body, True) if successful, (body, False) if the section was
    not found.
    """
    span = _find_section_span(body, old_name)
    if span is None:
        return body, False
    start, end = span
    # The heading line is just before `start`. Walk back to find it.
    line_start = body.rfind("\n", 0, start - 1) + 1
    # Replace heading line + content with the new ones.
    new = body[:line_start] + f"## {new_name}\n" + new_section_content + body[end:]
    return new, True


def _append_to_details(body: str, prose: str) -> tuple[str, list[str]]:
    """Append `prose` to the very start of `## Details` (so the folded Summary
    text appears as a leading paragraph in Details).
    """
    actions: list[str] = []
    span = _find_section_span(body, "Details")
    if span is None:
        return body, actions
    start, _end = span
    insertion = "\n" + prose.strip() + "\n\n"
    new_body = body[:start] + insertion + body[start:]
    actions.append(f"prepend folded Summary prose ({len(prose)} chars) into `## Details`")
    return new_body, actions


def _inject_stub_figure(body: str) -> tuple[str, list[str]]:
    """Insert a placeholder `## Figure` H2 (with `n/a` content) immediately
    before `## Details`. Used in v4-legacy conversion when the body has no
    Summary section containing an image.
    """
    actions: list[str] = []
    details_match = re.search(r"^## Details\s*$", body, re.MULTILINE)
    if not details_match:
        return body, actions
    insert_at = details_match.start()
    block = (
        "## Figure\n\n"
        "*Figure not applicable for this experiment (no headline plot was produced).*\n\n"
        "n/a\n\n"
    )
    new_body = body[:insert_at] + block + body[insert_at:]
    actions.append("inject `## Figure` stub before `## Details`")
    return new_body, actions


_REPRO_STUB = "## Reproducibility\n\n**Artifacts:** n/a\n\n**Compute:** n/a\n\n**Code:** n/a\n"


def _ensure_repro_section(body: str) -> tuple[str, list[str]]:
    """Inject `## Reproducibility` (with `n/a` subgroups) if it's missing.

    Inserts it right before the first non-target H2 that follows Details (e.g.
    `## Source issues`), or at the end of the body if no such trailing H2 exists.
    """
    actions: list[str] = []
    if _find_section_span(body, "Reproducibility") is not None:
        return body, actions
    # Find a trailing H2 (anything past Details that isn't in TARGET_H2).
    details_match = re.search(r"^## Details\s*$", body, re.MULTILINE)
    insert_at = len(body)
    if details_match:
        # Walk forward to the next H2 line after Details.
        after = body[details_match.end() :]
        next_h2 = re.search(r"^## (?!Details)\S", after, re.MULTILINE)
        if next_h2:
            insert_at = details_match.end() + next_h2.start()
    new_body = body[:insert_at].rstrip("\n") + "\n\n" + _REPRO_STUB + "\n" + body[insert_at:]
    actions.append("inject `## Reproducibility` stub with `n/a` subgroups")
    return new_body, actions


# ─── Top-level migration driver ───────────────────────────────────────────


def migrate_one(
    task_id: int,
    *,
    apply: bool,
    shape: str | None = None,
    verbose: bool = False,
) -> MigrateResult:
    """Migrate one task body. Returns a `MigrateResult` summary.

    Args:
        task_id: the awaiting_promotion task to migrate.
        apply: write changes back via `task_workflow.set_body`. If False
            (default), just report what would change.
        shape: optional override of the auto-classification; one of
            `"v4-to-new"` or `"conformant-failing"`. Useful for forcing a
            shape conversion against operator intuition.
        verbose: if True, populate `MigrateResult.diff_preview` with the
            first 60 lines of unified diff.
    """
    body_path = tw.find_task_path(task_id) / "body.md"
    fm, body = tw._read_body(body_path)
    cls = classify_body(body, fm=fm)

    # Verify-before status — pass the actual frontmatter so soft checks
    # that key off it (e.g. the Goal-of-experiment soft INFO check) see
    # the real values rather than an empty mapping.
    fm_text = _serialize_frontmatter(fm)
    overall_before, _ = vtb.verify_text(fm_text + body)
    verify_before = (
        "SKIP" if cls is BodyClass.LEGACY_HTML else ("PASS" if overall_before else "FAIL")
    )

    # Decide intent
    if shape == "v4-to-new":
        cls = BodyClass.V4_LEGACY
    elif shape == "conformant-failing":
        cls = BodyClass.CONFORMANT_FAILING

    actions: list[str] = []
    new_body = body
    needs_user = False
    needs_user_reason = ""

    if cls is BodyClass.PASS:
        return MigrateResult(task_id, cls, verify_before, "PASS")
    if cls is BodyClass.LEGACY_HTML:
        return MigrateResult(task_id, cls, "SKIP", "SKIP")
    if cls is BodyClass.UNKNOWN:
        return MigrateResult(
            task_id,
            cls,
            verify_before,
            verify_before,
            needs_user=True,
            needs_user_reason="body shape is neither v4-legacy nor four-H2 conformant",
        )

    snapshot_original = False
    if cls is BodyClass.V4_LEGACY:
        title = fm.get("title") if isinstance(fm, dict) else None
        new_body, conv_actions = convert_v4_to_target(body, title=title)
        actions.extend(conv_actions)
        snapshot_original = True
        # After shape conversion, attempt conformant-failing remediation too —
        # often the shape-converted body still misses Repro subgroups / cherry
        # labels / qual-data links.
        new_body, rem_actions = _conformant_remediate(new_body)
        actions.extend(rem_actions)
    elif cls is BodyClass.CONFORMANT_FAILING:
        new_body, rem_actions = _conformant_remediate(body)
        actions.extend(rem_actions)

    if new_body == body:
        # Nothing changed mechanically — the failing checks are not in the
        # mechanical-fix set (e.g. URL permanence, confidence rationale).
        return MigrateResult(
            task_id,
            cls,
            verify_before,
            verify_before,
            actions=actions,
            needs_user=True,
            needs_user_reason="failing checks lie outside the mechanical-fix set",
        )

    # Verify-after BEFORE we commit to writing — used to decide whether the
    # patch actually got us to PASS. Reuse the real frontmatter so the
    # Goal-of-experiment soft INFO check reflects on-disk state.
    overall_after_preview, _ = vtb.verify_text(fm_text + new_body)
    if not overall_after_preview:
        # Per plan §3 Phase E step 5: "If still failing, flag with
        # --needs-user and leave the body alone." Partial-credit patches are
        # discarded — the human takes over from the unmodified original.
        return MigrateResult(
            task_id,
            cls,
            verify_before,
            "FAIL",
            actions=actions,
            needs_user=True,
            needs_user_reason="mechanical patch insufficient — body left unchanged",
        )

    # Post-patch body PASSes the verifier — proceed with write or dry-run.
    diff_preview = ""
    if verbose:
        diff_preview = _unified_diff(body, new_body)[:6000]

    if apply:
        tw.set_body(task_id, new_body, snapshot_original=snapshot_original)
        verify_after_label = "PASS"
    else:
        verify_after_label = "DRY-PASS"

    return MigrateResult(
        task_id,
        cls,
        verify_before,
        verify_after_label,
        actions=actions,
        needs_user=needs_user,
        needs_user_reason=needs_user_reason,
        diff_preview=diff_preview,
    )


def _conformant_remediate(body: str) -> tuple[str, list[str]]:
    """Run all conformant-failing patches in order. Each is idempotent."""
    actions: list[str] = []
    body, a1 = remediate_repro_subgroups(body)
    actions.extend(a1)
    body, a2 = remediate_qual_data_link(body)
    actions.extend(a2)
    return body, actions


def _unified_diff(old: str, new: str) -> str:
    import difflib

    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            n=2,
        )
    )


def list_awaiting_promotion_ids() -> list[int]:
    """Return all task ids currently in `tasks/awaiting_promotion/`."""
    folder = tw.tasks_dir() / "awaiting_promotion"
    if not folder.is_dir():
        return []
    return sorted(int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit())


__all__ = [
    "BodyClass",
    "MigrateResult",
    "classify_body",
    "convert_v4_to_target",
    "list_awaiting_promotion_ids",
    "migrate_one",
    "remediate_qual_data_link",
    "remediate_repro_subgroups",
    "strip_v4_details_wrappers",
]


def _self_test() -> dict[str, Any]:  # pragma: no cover
    """Convenience smoke harness for interactive debugging."""
    out: dict[str, Any] = {}
    for tid in list_awaiting_promotion_ids():
        try:
            body_path = tw.find_task_path(tid) / "body.md"
            _, body = tw._read_body(body_path)
            out[tid] = classify_body(body).value
        except Exception as e:
            out[tid] = f"error: {e}"
    return out


if __name__ == "__main__":  # pragma: no cover
    import json as _json

    print(_json.dumps(_self_test(), indent=2))
