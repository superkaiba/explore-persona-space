"""task_workflow_migrate — migration helpers for `task.py migrate-body`.

Patches awaiting_promotion bodies into compliance with the 13-check
`verify_task_body.py` markdown spec. One patch mode:

Conformant-but-failing remediation — applied to bodies that already
carry the required H2 sections in order (the current 2-content-section
spec: Human TL;DR / TL;DR / Reproducibility, mirrored from
`verify_task_body.REQUIRED_H2_SECTIONS`) but fail one or more of the
content-level checks (Repro subgroups missing, cherry-picked label
missing on a sample-output fence, qualitative-data link missing on a
sample-output fence).

v4-legacy bodies (the pre-2026-05-13 `## TL;DR / ## Summary / ## Details
/ ## Source issues` shape) are still CLASSIFIED (`BodyClass.V4_LEGACY`)
but are no longer auto-converted: the old `convert_v4_to_target` chain
targeted the RETIRED four-H2 shape (TL;DR / Figure / Details /
Reproducibility), whose output always hard-FAILs the verifier's
stray-H2 check now that the target mirrors `REQUIRED_H2_SECTIONS`
(2-content-section spec, 2026-W22, task #454). The converter was retired
2026-06-09; `migrate_one` routes V4_LEGACY straight to `needs_user` and
leaves the body untouched — migrate manually per
`.claude/skills/clean-results/SPEC.md`.

Idempotency: every transformation is a string operation guarded by a
"would this change anything?" check; running `--apply` on an already-PASS
body is a no-op (no git diff).

The module exposes one entry point — `migrate_one(task_id, *, apply,
shape=None)` — which loads the task body, classifies it, runs the
remediation chain where applicable, optionally writes via
`task_workflow.set_body`, and returns a `MigrateResult` summary suitable
for `--report` rendering.
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
    CONFORMANT_FAILING = "conformant-failing"  # current required-H2 shape, but FAILs ≥1 check
    V4_LEGACY = "v4-legacy"  # ## TL;DR / ## Summary / ## Details / ## Source issues
    UNKNOWN = "unknown"  # neither of the above


V4_LEGACY_H2 = ("TL;DR", "Summary", "Details", "Source issues")
# The CURRENT conformant target shape — mirrored from the verifier so the
# classifier can never drift behind a spec migration again. Under the
# 2-content-section spec (2026-W22, task #454) this is
# ("Human TL;DR", "TL;DR", "Reproducibility"); the pre-W22 four-H2 shape
# (TL;DR / Figure / Details / Reproducibility) carries retired H2s that
# hard-FAIL verify check 2 and now classifies as UNKNOWN → needs_user
# (mechanical remediation cannot migrate retired-H2 content).
TARGET_H2 = tuple(vtb.REQUIRED_H2_SECTIONS)


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
    """The target shape: at least the required H2s (TARGET_H2) in the right order."""
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
            `"v4-to-new"` or `"conformant-failing"`. Forcing `"v4-to-new"`
            now deterministically reports `needs_user` (the v4 converter
            was retired 2026-06-09 — see module docstring).
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
            needs_user_reason=(
                "body shape is neither v4-legacy nor current-spec conformant "
                f"(required H2s: {list(TARGET_H2)})"
            ),
        )
    if cls is BodyClass.V4_LEGACY:
        # The v4 converter was retired 2026-06-09: it targeted the retired
        # four-H2 shape, whose output always hard-FAILs the verifier's
        # stray-H2 check under the 2-content-section spec. Route straight
        # to needs_user with the body untouched.
        return MigrateResult(
            task_id,
            cls,
            verify_before,
            verify_before,
            needs_user=True,
            needs_user_reason=(
                "v4-legacy shape predates the 2-content-section spec; "
                "auto-conversion was retired — migrate manually per "
                ".claude/skills/clean-results/SPEC.md"
            ),
        )

    # Only CONFORMANT_FAILING reaches the mechanical patch chain.
    new_body, actions = _conformant_remediate(body)

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
        tw.set_body(task_id, new_body, snapshot_original=False)
        verify_after_label = "PASS"
    else:
        verify_after_label = "DRY-PASS"

    return MigrateResult(
        task_id,
        cls,
        verify_before,
        verify_after_label,
        actions=actions,
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
    "list_awaiting_promotion_ids",
    "migrate_one",
    "remediate_qual_data_link",
    "remediate_repro_subgroups",
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
