"""Audit awaiting-promotion bodies for the body-discipline anti-patterns
identified during the 2026-05-08 mass-migration title pass.

Usage:

    # Audit a single task body (preferred for /issue Step 9a-bis):
    uv run python scripts/audit_clean_results_body_discipline.py --task <N>

    # Audit a local markdown file (e.g. an analyzer draft in /tmp):
    uv run python scripts/audit_clean_results_body_discipline.py /tmp/draft.md

    # Corpus-wide WARN-level H1-vs-frontmatter-title sync sweep (#1196):
    # one WARN row per sentinelled body whose H1 and frontmatter `title`
    # have drifted apart post-gate; WARN only — always exits 0.
    uv run python scripts/audit_clean_results_body_discipline.py --title-sync-sweep

    # Legacy bulk-inventory mode (no argument) — reads the pre-built
    # `.claude/cache/audit-2026-05-08/inventory.json` and writes the
    # findings markdown for every awaiting-promotion body listed there.

Bodies are NOT modified.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

OUT_DIR = Path(".claude/cache/audit-2026-05-08")
FINDINGS_PATH = OUT_DIR / "findings.md"
INVENTORY_PATH = OUT_DIR / "inventory.json"

PATTERNS: dict[str, tuple[str, str]] = {
    # name: (regex, plain-English description)
    "pre_reg": (
        r"pre-?registered|pre-?registration|(?<![a-z])pre-reg(?![a-z])|registered hypothesis"
        r"|registered alpha|\bas registered\b|fail at the gate|passed the gate"
        r"|gate-pre-?registered",
        "Pre-registration jargon ('pre-registered', 'as registered', "
        "'fail at the gate', 'gate-passed', etc.)",
    ),
    "verdict_caps": (
        # SUCCESS|FAILURE added for the #763 residual (#970): 'Under the
        # pre-set decision rule, SUCCESS was not met' escaped both pre_reg
        # (no 'as registered' bigram) and the original four-word alternation.
        # Case-sensitive scan (flags=0) keeps 'success'/'Success' clean.
        r"\b(?:REJECTED|INDETERMINATE|PASSED|EXCEEDING|SUCCESS|FAILURE)\b",
        "Pre-registration gate verdicts in CAPS "
        "(REJECTED / INDETERMINATE / PASSED / EXCEEDING / SUCCESS / FAILURE)",
    ),
    "effect_size_pp": (
        # The sign char classes include the typographic Unicode minus
        # (codepoint U+2212) alongside ASCII hyphen-minus and plus -- the same
        # blind spot the interval_inline fix below closed (#649): a negative
        # effect size rendered with U+2212 (Delta = MINUS 5pp) would otherwise
        # slip past. (RUF001/RUF003 noqa: the U+2212 in the pattern is the
        # literal char being matched, not an accidental homoglyph.)
        r"Δ[-−]?\d+\s*p?p|Δrate\s*=|Δ\s*=\s*[+-−]?\d+\s*(?:pp|%)",  # noqa: RUF001
        "Effect-size-in-percentage-points (Δ-Npp / Δrate / Δ = -Npp)",
    ),
    "interval_inline": (
        # Four alternatives, all banned in reader-facing PROSE (Lens 7):
        #   (1) `slope[low, high, ...]` — the original explicit slope form.
        #   (2) `[low, high]` followed by a CI verb/unit (excludes / includes /
        #       pp / % / ( / on) — the original trailing-token form.
        #   (3) the BARE bracketed-CI form `[+0.169, +0.437]` with NO trailing
        #       token — two signed/unsigned numbers separated by a comma inside
        #       brackets. Lens 7 names this the same banned construct as
        #       `value ± err`; the `±` regex missed it and the original
        #       trailing-token form let a bare pair slip past
        #       (caught only by the LM critic, incident #637). The two numbers
        #       need NOT both carry a decimal point (an integer-bound CI like
        #       `[1, 5]` is just as banned). Figure-caption blockquotes and the
        #       finding-internal "Why this test" definition line are exempted
        #       BEFORE the scan via `_strip_interval_inline_exempt_lines` — the
        #       Lens 7 carve-outs (chart annotations / the CI-as-test-definition
        #       sentence) — and GFM table cells via `_blank_table_rows` (the
        #       Reproducibility Parameters table + Data capsule tables carry
        #       interval forms legitimately).
        #   (4) the BOUND-FORM prose leak `(upper|lower) bound (+0.023)` /
        #       `upper bound = 0.0021` — a named CI/band endpoint stated as a
        #       number in prose with NO brackets. Lens 7 names this form
        #       explicitly (its own example: `upper bound = 0.0021`); all three
        #       prior alternatives require literal square brackets, so #952
        #       r1's "the interval's upper bound (+0.023) excludes the 0.05
        #       margin" reached the LM critic unflagged (incident #952 → fix
        #       #1015). Connectors are EXACTLY the two evidenced forms —
        #       `(num)` and `= num`; the `of`-connector is deliberately
        #       excluded ("an upper bound of 4 GPU-hours", "exceed the upper
        #       bound of 0.6" — budget / band-threshold prose, not a CI leak).
        #       The number must carry a sign OR a decimal point ("retry upper
        #       bound = 5" count-integers stay legal); the sign class includes
        #       U+2212 like the siblings. The [Uu]/[Ll] case pair exists
        #       because this category scans case-sensitively (flags=0), and
        #       the leading \b keeps embedded word tails out ("supper bound",
        #       "flower bound"). Singular `bound` only; no `~`/`≈` approx
        #       forms — zero corpus instances; the LM critic backstops those.
        #   The sign character class accepts ASCII hyphen-minus, ASCII plus,
        #   AND the typographic Unicode minus (codepoint U+2212) -- analyzers
        #   routinely render negative CI bounds with U+2212, so an ASCII-only
        #   sign class systematically missed half the sites of the construct
        #   this rule exists to catch (#649: two CIs whose lower bound used
        #   U+2212 slipped past while the ASCII-sign CIs in the SAME body were
        #   flagged). The U+2212 in each alternative below is the literal char
        #   being matched, not an accidental homoglyph -- hence the noqa.
        #   Band-notation carve-out: a bracketed integer interval immediately
        #   followed by a `nat` unit (`band-stop [5,12] nat`, an install /
        #   band-stop TARGET BAND in marker experiments) is NOT a credence
        #   interval of an estimate, so the bare-pair alternative excludes it
        #   via a `(?!\s*nat\b)` lookahead after the closing bracket. The
        #   trailing-token alternative is unaffected (it requires a CI verb /
        #   pp / % token, which a band never carries).
        r"slope\s*\[[-+−\d., ]+\]"  # noqa: RUF001
        r"|\[[-+−]?\d+\.\d+\s*,\s*[-+−]?\d+\.\d+\]\s*(?:excludes|includes|pp\b|%|\(|on\s)"  # noqa: RUF001
        r"|\[[-+−]?\d*\.?\d+\s*,\s*[-+−]?\d*\.?\d+\](?!\s*nat\b)"  # noqa: RUF001
        r"|\b(?:[Uu]pper|[Ll]ower)[\s-]bound\s*"
        r"(?:\(\s*(?:[-+−]?\d+\.\d+|[-+−]\d+)\s*\)|=\s*(?:[-+−]?\d+\.\d+|[-+−]\d+))",  # noqa: RUF001
        "Credence intervals as inline [low, high] or bound-form "
        "'(upper|lower) bound (+x)' / 'bound = x' in prose (banned)",
    ),
    "named_tests": (
        r"\bpaired t-test\b|\bFisher(?:'s)? exact\b|\bMann-Whitney\b"
        r"|\bbootstrap test\b|\bWilcoxon\b",
        "Named statistical tests in prose (paired t-test / Fisher / Mann-Whitney / Wilcoxon)",
    ),
    "h_symbols": (
        r"\bH_[a-zA-Z0-9]+\b|\bH[_-]?main\b",
        "Statistical-hypothesis symbols (H_a / H_0 / H_1 / H_main) without definition",
    ),
    "letter_labels": (
        r"\(\s*(?:[a-c]|[ivx]+)\s*\)\s+(?:slope|the|rate|sub-experiment)",
        "Anaphoric letter labels ('(a) slope ...', '(b) the ...') in prose",
    ),
    "bin_alpha": (
        r"\bBin\s+[A-E](?!\s*[a-z])",
        "Project-internal Bin labels (Bin A / Bin B / Bin C / Bin D / Bin E) "
        "without inline definition",
    ),
    "condition_labels": (
        # The U+2032 PRIME below is the literal char being matched (primed
        # condition labels like C1-prime), not an accidental homoglyph --
        # hence the noqa, mirroring the U+2212 entries above.
        r"\b[CcHhP][1-9](?:'|′)?"  # noqa: RUF001
        r"(?:\s*(?:condition|control|completion|coefficient|hypothesis|test"
        r"|sub-?(?:claim|experiment|hypothesis)))?(?![a-zA-Z0-9_])",
        "Project-internal condition/hypothesis labels (C1/C2/C3, H1/H2/H3, P1/P2/P3 "
        "with optional prime) — replace with named conditions inline",
    ),
    "cell_tags": (
        # Per-cell / per-condition / per-judge plan-internal tags:
        #   BS_E0, BS_E_42, Z_assistant, Z_villain (uppercase + underscore + alphanum)
        #   B0 / B1 as standalone (not "B0:" inside table headers — check context)
        #   G6 / G0a / G2-escalation (judge / gate labels)
        #   M1 / M2 (extraction-method labels — only flag when paired with "cosine"/"cell"/"method")
        #   "Method A" / "Method B" (extraction-method labels — uppercase Method + capital letter)
        r"\bBS_E[0-9A-Za-z_]*|\bZ_[a-zA-Z_]+|\b[Gg][0-9]+[a-c]?\b(?=\s|:|\.|,|$)|\bMethod\s+[AB]\b|\b[Mm][1-9]\b(?=\s+(?:cosine|cell|mean|extraction|method|sub-experiment))",
        "Plan-internal per-cell / extraction-method / judge / gate tags (BS_E*, Z_*, G*, "
        "Method A/B, M1) — replace with plain English; tags go in "
        "<details>Setup details</details>",
    ),
    "experimental_arm": (
        # "arm" / "arms" used as a project-internal experiment-strand label.
        # Excludes legitimate uses: "arm rest", "human arm", "arm yourself".
        # Triggers on: "<adj>-arm", "the <adj> arm", "behavioral arm", "geometric arm",
        # "five arms", "experimental arm(s)".
        r"\b(?:experimental|behavioral|geometric|reverse-?order|forward-?order|length(?:[-/\s]style)?|full[-\s]?param(?:eter)?|LoRA)\s+arms?\b|\b(?:five|four|three|two)\s+arms?\b|\bexperimental\s+arms?\b|\b(?:the|a)\s+(?:behavioral|geometric|reverse-?order|forward-?order|length(?:[-/\s]style)?|full[-\s]?param(?:eter)?|LoRA)\s+arm\b",
        "Project-internal experiment-strand 'arm' label — describe what was done, "
        "not the strand's name",
    ),
    "bare_method_acronym": (
        r"\b(?:GCG|PAIR|EvoPrompt|nanoGCG)\b",
        "Methodology acronyms (GCG / PAIR / EvoPrompt / nanoGCG) — flag for definition check",
    ),
    "stats_acronyms": (
        r"\b(?:OLS|MLE|ANOVA|ROC)\b",
        "Statistical acronyms (OLS / MLE / ANOVA / ROC) without inline definition",
    ),
    "auc_bare": (
        r"\bAUC\s*=\s*0\.\d+",
        "AUC = X.XX values — verify each is paired with what it's computed on",
    ),
    "post_hoc_phrasing": (
        r"\bpost-hoc\b|\bex post\b",
        "'post-hoc' / 'ex post' — academic-paper register; usually droppable",
    ),
    "math_notation": (
        # Identifier with caret-superscript (R^P2, R_B^P2, R_BgivenA^P2),
        # OR identifier with two-segment underscore-subscript that is itself
        # capitalized math notation (R_BgivenA, P_TopK). The second arm is
        # narrower than rule 10's h_symbols catch (which is H_*-specific) so
        # we only flag CamelCase / multi-letter subscripts that look like
        # math identifiers — not file paths or `eval_results/foo` variables.
        r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*\^[A-Za-z0-9_*+\-]+|\b[A-Z]_[A-Z][A-Za-z]{2,}\b",
        "Math-style subscript/superscript notation in prose (R_BgivenA^P2, R^P2, P_TopK) "
        "— markdown doesn't render these",
    ),
    "bit_byte_identical": (
        # "byte identical" / "byte-identical" AND the same-family "bit
        # identical" / "bit-identical" anywhere in body prose. Banned
        # 2026-W22 (task #454, byte form) — the phrase reads as AI-slop in
        # research writing. The `bit` variant was added 2026-W25 (task #642:
        # the body carried `bit-identical` AND `byte-identical`, but the
        # byte-only regex flagged only the latter; the clean-result-critic
        # Lens 6 bans both forms as the same voice violation, so the audit
        # must catch both under one rule). Use plain English: "the two files
        # matched exactly", "every byte agreed", "no diff between the runs".
        r"\b(?:byte|bit)[\s-]identical\b",
        (
            "Use plain English ('the two files matched exactly', 'every byte agreed', "
            "'no diff') instead of 'byte identical' / 'byte-identical' / 'bit identical' / "
            "'bit-identical' — the phrase reads as AI-slop in research prose"
        ),
    ),
}


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def list_awaiting_promotion() -> list[dict]:
    """Read pre-built inventory.json (from bash paginator) — Python's gh
    GraphQL pagination chokes on cursors with certain characters."""
    return json.loads(INVENTORY_PATH.read_text())


def strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block (``---`` … ``---``).

    The anti-pattern audit is about PROSE discipline in the body. YAML
    frontmatter carries structured metadata — e.g. ``relates_to: [d1, d3,
    h2]`` open-question IDs — that is not prose and must not be scanned
    for project-internal-label patterns (``h2`` is an open-question ID,
    not a ``H2`` hypothesis label).
    """
    if text.startswith("---"):
        m = re.match(r"^---\n.*?\n---\n", text, flags=re.DOTALL)
        if m:
            return text[m.end() :]
    return text


def strip_code(text: str) -> str:
    """Remove fenced code blocks and inline-backtick spans."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", "", text)
    return text


def strip_fenced_code_only(text: str) -> str:
    """Remove fenced ```...``` code blocks but KEEP inline-backtick spans.

    Companion to :func:`strip_code`. The `interval_inline` scan source uses
    THIS helper (not `strip_code`) so an inline-backtick-wrapped bracketed
    CI in reader-facing prose (e.g. ``CI `[-0.295, +0.083]` ``) is still
    seen by the bracketed-CI regex — `strip_code`'s inline-backtick blanking
    (the `` `[^`\\n]*` `` substitution) otherwise removes it before the scan
    (the #667 line-166 gap). Sample-completion text in FENCED code blocks
    is still stripped, so this does not create false positives from verbatim
    bracketed expressions inside fenced examples. Every OTHER audit category
    keeps using `strip_code` (full inline+fenced strip) — inline-backtick
    `C1`/`D1` references are a legitimate accepted form for `condition_labels`
    and we deliberately do not widen its exposure here.
    """
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


# GFM table delimiter row: `|---|---|`, `:--|:-:|--:`, `---|---`, etc.
# Mirrors `_TABLE_DELIM_RE` in `verify_task_body.py`: at least TWO cells of
# dashes (with optional leading/trailing `|` and optional `:` alignment
# markers) separated by an internal `|`. The internal `|` is mandatory; it
# is what distinguishes a real multi-column GFM table delimiter from a
# bare `---` thematic break or setext-style H2 underline.
_TABLE_DELIM_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*\|\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)*\|?\s*$"
)


def _table_row_line_indices(lines: list[str]) -> set[int]:
    """Return the indices of lines that belong to a GFM table block.

    A GFM table is a header row (a `|`-containing line) IMMEDIATELY
    followed by a delimiter row (`_TABLE_DELIM_RE`), then a contiguous
    run of `|`-containing body rows until a blank line or a non-pipe
    line. Mirrors `verify_task_body.py::_table_row_line_indices` — the
    canonical table-cell detector used by check 14 — so the audit and
    verifier carry the same table-block definition. A lone prose line
    that happens to carry a `|` (e.g. `log p(x | y)` in a paragraph)
    is NOT a table row — it lacks the required delimiter neighbor.

    Lines inside fenced code blocks are excluded (callers also strip
    fences via `strip_code`, but we guard here too so the delimiter
    scan can't be tricked by a `|---|` shown inside a code fence).

    Used by `audit_body` to exempt table cells from prose-only audit
    categories (`interval_inline`, `condition_labels`). The
    clean-result-critic Lens 7 spec (`.claude/agents/clean-result-critic.md`)
    scopes the bracketed-CI ban to TL;DR / Findings / Reproducibility
    PROSE — table cells in the Reproducibility Parameters table
    legitimately carry interval forms like `mc_ci = [0.236, 0.252]`.
    The `condition_labels` rule's purpose is symmetrically to catch
    BARE codes in narrative prose where the reader has no lookup; a
    persona-ID lookup table whose entire purpose IS defining `C1` /
    `D1` is not the target of that rule.
    """
    table_lines: set[int] = set()
    in_fence = False
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        if (
            "|" in stripped
            and not _TABLE_DELIM_RE.match(stripped)
            and i + 1 < n
            and _TABLE_DELIM_RE.match(lines[i + 1].strip())
        ):
            table_lines.add(i)  # header
            table_lines.add(i + 1)  # delimiter
            j = i + 2
            while j < n:
                row = lines[j].strip()
                if row == "" or "|" not in row:
                    break
                if row.startswith("```") or row.startswith("~~~"):
                    break
                table_lines.add(j)
                j += 1
            i = j
            continue
        i += 1
    return table_lines


def _blank_table_rows(text: str) -> str:
    """Return a copy of `text` with every GFM table-row line blanked.

    "Blanked" means the line's content is replaced with an empty
    string (the `\n` is preserved). Used by `audit_body` to produce a
    table-cell-exempt scan source for the prose-only categories
    (`interval_inline`, `condition_labels`), and by
    `_restrict_pre_reg_to_prose_sections` for the v4-body `pre_reg`
    scope (whole-body-minus-tables). Non-exempt categories keep
    scanning the unblanked text — `bit_byte_identical`, `named_tests`,
    `letter_labels`, etc. are not prose-vs-table sensitive, and we
    don't want to silently widen the audit's exemption surface.
    """
    lines = text.splitlines()
    table_idx = _table_row_line_indices(lines)
    return "\n".join("" if i in table_idx else line for i, line in enumerate(lines))


# Audit categories whose regex hits inside a real GFM table cell are
# spec-compliant and must be suppressed. Lens 7 (clean-result-critic)
# scopes the bracketed-CI ban to PROSE surfaces; the Parameters table
# in `## Reproducibility` legitimately carries interval forms like
# `mc_ci = [0.236, 0.252]`. The persona-ID lookup table inside
# `### What I ran` legitimately carries `C1` / `D1` codes whose
# definition IS the table — the `condition_labels` rule targets BARE
# codes in narrative prose where the reader has no lookup. Other
# categories (`bit_byte_identical`, `named_tests`, ...) keep
# firing on table cells — the prose-vs-table distinction is not
# load-bearing for those. `pre_reg` is a v4-only, function-local
# exception: on v4 bodies `_restrict_pre_reg_to_prose_sections` blanks
# table rows itself (membership of this frozenset is UNCHANGED — routing
# `pre_reg` through it would leak the table exemption to v2/legacy
# bodies, and `audit_body` dispatches `pre_reg` before this set anyway).
_TABLE_CELL_EXEMPT_CATEGORIES: frozenset[str] = frozenset({"interval_inline", "condition_labels"})


# Lens 7 (clean-result-critic) scopes the bracketed-CI ban to reader-facing
# PROSE — Takeaways bullets and finding setup/read paragraphs — and names two
# carve-outs: (1) chart annotations / figure captions (the CI summarises a
# plotted distribution, like a chart error bar), and (2) the finding-internal
# "Why this test" definition sentence that explicitly names the CI as part of
# the test definition. `verify_task_body.py` already treats blockquote (`>`)
# lines as figure-caption reference material, not prose; mirror that here.
# Both carve-outs are blanked BEFORE the `interval_inline` scan (and ONLY for
# that category — the figure-caption / Why-this-test distinction is not
# load-bearing for `condition_labels` or any other rule, so we don't widen
# their exemption surface).
_WHY_THIS_TEST_RE = re.compile(r"why\s+this\s+test", re.IGNORECASE)


def _strip_interval_inline_exempt_lines(text: str) -> str:
    """Blank figure-caption blockquote lines and the finding-internal
    "Why this test" definition line — the two Lens 7 carve-outs for the
    bracketed-CI / `interval_inline` category.

    "Blanked" means the line's content becomes an empty string (the
    trailing `\n` is preserved, so line offsets in sample output stay
    stable). A blockquote line is any line whose first non-whitespace
    character is `>` (the v3 figure-caption form `> **Figure.** ...`);
    a "Why this test" line is any line containing that phrase
    (case-insensitive), the inline definition sentence Lens 7 exempts.

    Scoped to `interval_inline` only via the caller in :func:`audit_body`;
    other categories keep scanning the unblanked text.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">") or _WHY_THIS_TEST_RE.search(line):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


# The `## Reproducibility` `**Context:**` provenance row (SPEC.md
# § `**Context:**` row; verify_task_body.py check 17) requires the
# originating user prompt / follow-up scope note be carried forward
# VERBATIM — never paraphrased, trimmed, or typo-fixed. Verbatim
# preservation and the prose anti-pattern scan are mutually unsatisfiable
# on that quote (task #597: a scope note opening with "PRE-REGISTERED"
# tripped the `pre_reg` pattern; task #651: an originating prompt
# containing "post-hoc" tripped `post_hoc_phrasing`), so the verbatim
# prompt is exempt from the scan in BOTH the forms the corpus uses:
#   (1) a following `>` blockquote (SPEC.md's recommended shape, e.g.
#       #638 / #537 / #627) — blanked by the `>` branch below; AND
#   (2) the prompt carried INLINE on the `- Originating prompt: "..."`
#       sub-bullet (and any wrapped continuation lines), the form #651
#       and #640 / #610 use — blanked by the `in_origin_prompt` branch.
# Non-prompt prose inside the block (the Created / Follow-up to bullets)
# is still scanned.
_CONTEXT_LABEL_RE = re.compile(r"^(?:[-*]\s+)?\*\*\s*Context\s*:?\s*\*\*")
_BOLD_LABEL_RE = re.compile(r"^(?:[-*]\s+)?\*\*\s*([^*\n]+?)\s*:?\s*\*\*")
# SPEC.md names exactly three Context sub-bullets; a boldface label
# outside this set (e.g. **Compute:**, **Code:**) starts a sibling row
# and ends the block. Plain (non-bold) sub-bullets never match
# _BOLD_LABEL_RE, so they keep the block open without this whitelist.
_CONTEXT_SUB_LABELS = ("created", "follow-up to", "originating prompt")
# The `Originating prompt` sub-bullet in EITHER form: a list item whose
# label is "originating prompt(s)[, verbatim]", optionally boldfaced.
# Matches both `- Originating prompt: "..."` (plain, #651) and
# `- **Originating prompt(s), verbatim:** ...` (bold, #640 / #610). The
# verbatim prompt text follows on the same line and/or wrapped
# continuation lines; the `in_origin_prompt` walker blanks all of it.
_ORIGIN_PROMPT_SUBLABEL_RE = re.compile(r"^[-*]\s+(?:\*\*\s*)?originating\s+prompt", re.IGNORECASE)
_LIST_OR_BOLD_RE = re.compile(r"^(?:[-*]\s|\*\*)")


def strip_context_blockquotes(text: str) -> str:
    """Drop the verbatim originating-prompt quote inside the `**Context:**`
    provenance block — in both the blockquote and inline forms.

    The block runs from the `**Context:**` label to the next markdown
    heading or the next boldface row label that is not one of the
    Context sub-bullets (Created / run, Follow-up to, Originating
    prompt(s)), or EOF. Within the block:

    - `>` blockquote lines are blanked (SPEC.md's recommended verbatim
      shape — incident #597).
    - The `- Originating prompt …` sub-bullet AND its wrapped
      continuation lines are blanked, so an inline verbatim prompt is
      exempt too (incident #651: an inline prompt containing "post-hoc"
      collided with `post_hoc_phrasing`). The inline-prompt run ends at
      the next sibling list item / bold label / heading / blank line, or
      the Context-block end — whichever comes first.

    If a boundary is mis-detected the failure mode is the pre-fix
    behavior (the quote gets scanned) — never a silently widened
    exemption.
    """
    out_lines: list[str] = []
    in_context = False
    in_origin_prompt = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_context:
            # An inline-prompt continuation run ends at a blank line, a
            # sibling list item / bold label, or a heading.
            if in_origin_prompt and (
                stripped == "" or stripped.startswith("#") or _LIST_OR_BOLD_RE.match(stripped)
            ):
                in_origin_prompt = False
            label_match = _BOLD_LABEL_RE.match(stripped)
            if stripped.startswith("#") or (
                label_match
                and not any(
                    label_match.group(1).lower().startswith(sub) for sub in _CONTEXT_SUB_LABELS
                )
            ):
                in_context = False
                in_origin_prompt = False
            elif _ORIGIN_PROMPT_SUBLABEL_RE.match(stripped):
                in_origin_prompt = True
                continue  # inline verbatim prompt label + same-line quote — exempt
            elif in_origin_prompt or stripped.startswith(">"):
                continue  # verbatim provenance quote (inline run or blockquote) — exempt
        if not in_context and _CONTEXT_LABEL_RE.match(stripped):
            in_context = True
        out_lines.append(line)
    return "\n".join(out_lines)


_H2_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")


# H2 sections whose `<details>` example blocks are exempt from the prose
# scan: `## Data` (v3 spec — `### Trained on` / `### Evaluated with` /
# `### Generated` example blocks) and `## Methodology` (v4 spec — the
# `**Sample training/evaluation data + completions:**` slot;
# `.claude/skills/clean-results/SPEC.md` mandates verbatim example
# rows/completions wrapped in `<details>` or a fenced code block, and
# fenced blocks are already stripped globally). Exact lowercase H2-title
# match, mirroring the original `## Data`-only equality check (#1171).
_DETAILS_EXEMPT_H2_TITLES: frozenset[str] = frozenset({"data", "methodology"})


def strip_data_example_blocks(text: str) -> str:
    """Drop `<details>...</details>` example blocks inside the `## Data`
    (v3) and `## Methodology` (v4) sections.

    The v3 clean-result spec MANDATES verbatim training rows / eval
    probes / sample completions inside `## Data` (`### Trained on` /
    `### Evaluated with` / `### Generated`), carried in `<details>`
    example blocks. Those verbatim rows routinely contain strings the
    prose anti-pattern scan flags as project-internal condition codes
    (`C1`, `H2`, `M1`, `BS_E0`, …) with NO reword option — the same
    verbatim-content conflict the `**Context:**` blockquote carve-out
    fixed (incident #597). The author cannot paraphrase a row that is
    required to be verbatim, so example blocks inside `## Data` are
    exempt from the scan.

    The v4 spec (`<!-- clean-result-v4 -->`) moved the verbatim sample
    rows to `## Methodology` → `**Sample training/evaluation data +
    completions:**`, carried in the same `<details>` / fenced forms
    (SPEC.md), so the v4 section gets the identical exemption (#1171).
    Methodology PROSE (`**Design:**` / `**Training:**` /
    `**Evaluation:**` lines outside a `<details>` block) stays scanned.

    Mechanism mirrors :func:`strip_context_blockquotes`: a stateful
    line walker that drops only lines inside a `<details>` block while
    the cursor is inside an exempt section
    (`_DETAILS_EXEMPT_H2_TITLES`). Fenced code blocks (the other
    example-block form) are already removed globally by
    :func:`strip_code`, so this only needs to handle `<details>` blocks.

    An exempt section runs from its `## ` H2 to the next `## ` H2
    (typically `## Reproducibility` / `## Results`) or EOF. If a
    `</details>` close is never seen before the section ends, the
    block-drop ends with the section — a mis-detected boundary degrades
    to the pre-fix behavior (the lines get scanned), never a silently
    widened exemption.
    """
    out_lines: list[str] = []
    in_exempt_section = False
    in_details = False
    for line in text.splitlines():
        stripped = line.strip()
        h2 = _H2_RE.match(stripped)
        if h2:
            # Any H2 ends an exempt section (and any in-flight details
            # drop); re-enter only when the new H2 is `## Data` (v3) or
            # `## Methodology` (v4) itself.
            in_exempt_section = h2.group("title").strip().lower() in _DETAILS_EXEMPT_H2_TITLES
            in_details = False
            out_lines.append(line)
            continue
        if in_exempt_section:
            lowered = stripped.lower()
            if not in_details and lowered.startswith("<details"):
                in_details = True
                continue  # drop the verbatim example block
            if in_details:
                if "</details>" in lowered:
                    in_details = False
                continue  # drop every line inside the example block
        out_lines.append(line)
    return "\n".join(out_lines)


# Lens 7 (clean-result-critic) scopes the PRE-REGISTRATION-mention ban to the
# three reader-facing v3 prose sections ONLY — `## Takeaways` / `## What I ran`
# / `## Findings` — and explicitly permits pre-reg threshold values to sit in
# the parameters table (`.claude/agents/clean-result-critic.md` Lens 7:
# "Pre-registration mentions … in `## Takeaways` / `## What I ran` /
# `## Findings` prose. Pre-reg threshold values can sit in the parameters
# table."). The `pre_reg` regex otherwise scans the whole body, so a
# procedural "dropped pre-registered" sentence in `## Data` / `## Reproducibility`
# prose — spec-permitted there — fires a FALSE positive the critic must
# hand-adjudicate every round (incident #623: an `## Data → ### Evaluated with`
# pre-reg mention tripped the audit although Lens 7 exempts that section).
# Under the v4 spec (`<!-- clean-result-v4 -->`) Lens 7 instead bans pre-reg
# mentions in ALL FOUR H2s' prose (`## Takeaways` / `## Goal` /
# `## Methodology` / `## Results`) and permits threshold values ONLY in the
# Methodology Training hyperparameter table — implemented in
# `_restrict_pre_reg_to_prose_sections` as a whole-body scan with GFM table
# rows blanked (ALL positively-detected tables, deliberately wider than the
# named Training table; the LM clean-result-critic backstops non-Methodology
# tables).
# This carve-out is `pre_reg`-only; the other Lens 7 sub-categories (named
# tests, power analyses, inline `value ± err`) are NOT section-scoped in the
# spec, so we do not widen their exemption surface.
_PRE_REG_PROSE_SECTIONS = ("takeaways", "what i ran", "findings")


def _restrict_pre_reg_to_prose_sections(body: str, text: str) -> str:
    """Return the `pre_reg` scan source for `text`, scoped per the body's
    clean-result generation (three regimes):

    - v4 (`<!-- clean-result-v4 -->`): whole-body scan with every
      positively-detected GFM table-row line blanked. Lens 7 bans pre-reg
      mentions in ALL FOUR v4 H2s' prose (`## Takeaways` / `## Goal` /
      `## Methodology` / `## Results`) and the ONLY surface it explicitly
      permits is the Methodology Training hyperparameter table; since the
      four sections are nearly the whole v4 body (stray content H2s are a
      `verify_task_body.py` hard FAIL), the scope is whole-body-minus-tables.
      NOTE the table exemption is deliberately WIDER than Lens 7's letter:
      it blanks ALL positively-detected GFM table rows, not only the
      Methodology Training table — the LM clean-result-critic remains the
      backstop for a pre-reg mention smuggled into a non-Methodology table.
      The verbatim originating prompt in the `**Context:**` footer is
      already exempt for every generation via `strip_context_blockquotes`
      (#597/#651); the rest of the footer stays scanned deliberately (no
      incident; the conservative direction).
    - v3 (`<!-- clean-result-v3 -->`): blank every line OUTSIDE the three
      Lens 7 v3 prose sections (`## Takeaways` / `## What I ran` /
      `## Findings`), UNCHANGED (incident #623).
    - v2 / legacy / unstructured (which use `## AI TL;DR` / `## Human
      TL;DR` / `## TL;DR` / `## Details` H2 names): `text` is returned
      unchanged, so the prior whole-body `pre_reg` behavior is preserved
      verbatim and we never silently blank an entire legacy body's prose.

    `text` is the already-cleaned scan source (frontmatter / code / Context
    blockquote / Data example blocks stripped); `body` is the raw body, used
    only for the sentinel gates. "Blanked" means the line content becomes
    an empty string (the trailing `\n` is preserved so sample-output offsets
    stay stable).

    Degradation property: `_blank_table_rows` blanks only lines
    `_table_row_line_indices` positively identifies as GFM pipe-table rows
    (behavior pinned by the existing interval_inline / condition_labels
    tests), so a misshapen table degrades to its rows being SCANNED (a
    hand-adjudicated false positive) — never to a silently widened
    exemption; every PROSE surface not explicitly permitted keeps firing,
    and a mis-detected v3 section boundary likewise degrades to scanning a
    line that should have been blanked. v4 is checked BEFORE v3 because
    the v4 sentinel declares the governing spec for the body — NOT because
    v4 scans a strict superset (it does not: a table row inside
    `## Takeaways` / `## Findings` is scanned by the v3 walker yet blanked
    by the v4 branch); on a malformed dual-sentinel body the v4 branch at
    least keeps every prose surface in scope.
    """
    if "<!-- clean-result-v4 -->" in body:
        return _blank_table_rows(text)
    if "<!-- clean-result-v3 -->" not in body:
        return text
    out: list[str] = []
    in_prose_section = False
    for line in text.splitlines():
        h2 = _H2_RE.match(line.strip())
        if h2:
            in_prose_section = h2.group("title").strip().lower() in _PRE_REG_PROSE_SECTIONS
            # The H2 heading line itself carries no scannable prose; blank
            # it whether or not it opens a prose section.
            out.append("")
            continue
        out.append(line if in_prose_section else "")
    return "\n".join(out)


def is_v2(body: str) -> bool:
    """Return True when a body is treated as a "current spec" body for
    the legacy bulk-inventory audit.

    Historically this matched the retired AI TL;DR / AI Summary
    four-H2 shape. Under the 2-content-section nested-design (v2) spec
    (`.claude/skills/clean-results/SPEC.md`, migrated 2026-W22 task
    #454 + nested-TL;DR adoption forward-only) and the five-flat-H2
    (v3) redesign (2026-W24, sentinel `<!-- clean-result-v3 -->`),
    "current spec" now means ANY of:

    - The v4 sentinel `<!-- clean-result-v4 -->` is present (the
      current prescriptive shape, four-flat-H2, migrated 2026-W26:
      Takeaways / Goal / Methodology / Results); OR
    - The v3 sentinel `<!-- clean-result-v3 -->` is present (prior
      prescriptive shape: Takeaways / What I ran / Findings /
      Data / Reproducibility); OR
    - The nested-design (v2) sentinel `<!-- clean-result-v2 -->` is
      present in the body (prior prescriptive shape); OR
    - The body carries `## Human TL;DR` AND `## TL;DR` AND
      `## Reproducibility` H2s (the post-#454 flat shape, still
      promotable for legacy bodies); OR
    - Legacy fallback: the retired "AI TL;DR" / "AI Summary" markers
      (kept so the bulk-inventory audit doesn't drop pre-#454 bodies
      from consideration).

    This is a coarse "should I audit this body's prose" gate consulted
    ONLY on the bulk-inventory path (`_run_legacy_bulk_inventory`); the
    live pipeline paths (`--task <N>` at /issue Step 9a-bis, the
    explicit-file path for analyzer drafts) audit UNCONDITIONALLY and do
    NOT consult this gate. It is NOT a structural verifier —
    `scripts/verify_task_body.py` is the authoritative mechanical gate.
    """
    if "<!-- clean-result-v4 -->" in body:
        return True
    if "<!-- clean-result-v3 -->" in body:
        return True
    if "<!-- clean-result-v2 -->" in body:
        return True
    if "## Human TL;DR" in body and "## TL;DR" in body and "## Reproducibility" in body:
        return True
    # Legacy AI TL;DR / AI Summary fallback (pre-#454 shape).
    return "## AI TL;DR (human reviewed)" in body or (
        "## AI TL;DR" in body and "## AI Summary" in body
    )


def audit_body(body: str) -> dict[str, list[str]]:
    """Scan `body` for prose anti-patterns under the categories in
    `PATTERNS`. Returns a dict of category-name -> up-to-5 sample hits.

    Table-cell exemption: categories in `_TABLE_CELL_EXEMPT_CATEGORIES`
    (`interval_inline`, `condition_labels`) scan a copy of the cleaned
    text with every GFM table-row line blanked. This mirrors the spec
    — the clean-result-critic Lens 7 rule scopes the bracketed-CI ban
    to PROSE, and the Reproducibility Parameters table legitimately
    carries interval forms (`mc_ci = [0.236, 0.252]`). The persona-ID
    lookup table inside `### What I ran` legitimately carries `C1` /
    `D1` codes whose definition IS the table. All other categories
    keep scanning the unblanked text — the prose-vs-table distinction
    is not load-bearing for `bit_byte_identical`, `named_tests`,
    `letter_labels`, etc. (`pre_reg` is the one exception: on v4 bodies
    its scan source ALSO blanks table rows, routed function-locally
    through `_restrict_pre_reg_to_prose_sections` below — NOT through
    `_TABLE_CELL_EXEMPT_CATEGORIES`, whose membership is unchanged).

    `interval_inline` additionally blanks figure-caption blockquotes and
    the finding-internal "Why this test" line (Lens 7's two carve-outs)
    via `_strip_interval_inline_exempt_lines` — bracketed bounds in a
    chart caption or a CI-as-test-definition sentence are spec-compliant.

    `pre_reg` scans a generation-scoped source via
    `_restrict_pre_reg_to_prose_sections` (three regimes): v4 bodies scan
    the whole body with GFM table rows blanked (Lens 7 bans the mention in
    all four v4 H2s' prose and permits threshold values in the Methodology
    Training hyperparameter table); v3 bodies scan ONLY the three Lens 7
    v3 prose sections (`## Takeaways` / `## What I ran` / `## Findings`) —
    a spec-permitted procedural "dropped pre-registered" sentence in
    `## Data` / `## Reproducibility` prose no longer fires a false
    positive (incident #623); v2 / legacy bodies keep the prior whole-body
    `pre_reg` behavior.
    """
    findings: dict[str, list[str]] = {}
    cleaned = strip_code(
        strip_data_example_blocks(strip_context_blockquotes(strip_frontmatter(body)))
    )
    cleaned_table_blanked = _blank_table_rows(cleaned)
    # `interval_inline` uses `strip_fenced_code_only` (NOT `strip_code`) so an
    # inline-backtick-wrapped bracketed CI in prose (``CI `[-0.295, +0.083]` ``,
    # the #667 line-166 gap) is still seen — `strip_code` blanks inline-backtick
    # spans, hiding the CI before the scan. The SAME downstream exemptions still
    # apply (Data/Methodology `<details>` / Context block stripped by the inner chain; table
    # rows blanked by `_blank_table_rows`; figure-caption + Why-this-test lines
    # blanked by `_strip_interval_inline_exempt_lines`). Fenced code blocks are
    # still stripped, so verbatim bracketed expressions in fenced examples do
    # not false-positive. Every OTHER category keeps `cleaned` / `strip_code`.
    interval_cleaned = strip_fenced_code_only(
        strip_data_example_blocks(strip_context_blockquotes(strip_frontmatter(body)))
    )
    interval_scan_source = _strip_interval_inline_exempt_lines(_blank_table_rows(interval_cleaned))
    # `pre_reg` scans a generation-scoped source: v4 = whole body minus
    # table rows; v3 = the three Lens 7 prose sections; v2/legacy = whole body.
    pre_reg_scan_source = _restrict_pre_reg_to_prose_sections(body, cleaned)
    for name, (pattern, _) in PATTERNS.items():
        if name == "interval_inline":
            scan_source = interval_scan_source
        elif name == "pre_reg":
            scan_source = pre_reg_scan_source
        elif name in _TABLE_CELL_EXEMPT_CATEGORIES:
            scan_source = cleaned_table_blanked
        else:
            scan_source = cleaned
        flags = re.IGNORECASE if name == "pre_reg" else 0
        matches = list(re.finditer(pattern, scan_source, flags))
        if matches:
            findings[name] = [m.group(0) for m in matches[:5]]
    return findings


def _resolve_task_body_path(task_number: int) -> Path:
    """Resolve `tasks/<status>/<task_number>/body.md` via the
    task_workflow helper (same lookup used by `verify_task_body.py`)."""
    from explore_persona_space.task_workflow import find_task_path

    return find_task_path(task_number) / "body.md"


_RE_FM_PAPER = re.compile(r"(?im)^paper\s*:\s*(?:true|'true'|\"true\")\s*$")


def _is_paper_stub(body: str) -> bool:
    """True when the body's leading frontmatter carries `paper: true`.

    A `paper: true` task's body.md is a thin paper-stub (H1 + abstract + paper
    link); its canonical clean-result is the LaTeX paper under
    docs/papers/issue_<N>/, audited by scripts/verify_paper.py — the markdown
    body-discipline anti-patterns do NOT apply.
    """
    if not body.startswith("---"):
        return False
    end = body.find("\n---", 3)
    head = body[3:end] if end != -1 else body
    return bool(_RE_FM_PAPER.search(head))


def _load_verify_task_body():
    """Import scripts/verify_task_body.py as a sibling-script module.

    Robust to BOTH launch modes: `uv run python scripts/audit_...py`
    (scripts/ is already sys.path[0]) and the test file's
    importlib.spec_from_file_location loading (which does NOT put
    scripts/ on sys.path). Follows the established sibling-import
    convention (_rest_backfill.py:19, backfill_artifact_registry.py:34).
    Lazy — called inside the check helper, cached by Python's module
    cache — so the auditor's own import stays light and the legacy /
    frontmatter-less paths pay nothing.
    """
    import importlib
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("verify_task_body")


def h1_title_sync_warn(text: str) -> str | None:
    """WARN-severity corpus mirror of verify_task_body.py's gate-time
    `check_h1_matches_frontmatter_title` (#1110 gate check; #1196 corpus
    surface). Delegates the ENTIRE comparison — the fence-aware sentinel
    gate (v4/v3/v2-nested), the frontmatter parse, the
    whitespace-collapse-only normalization, and the missing-fm-title /
    missing-H1 anomaly branches — to the gate check, then flattens its
    severity: any flagged outcome (gate FAIL on v4, gate WARN on
    grandfathered v3/v2) returns the check's detail string; an in-sync
    or out-of-scope body returns None. WARN only: callers never let this
    affect the exit code — post-gate remediation is a human call.
    """
    vtb = _load_verify_task_body()
    fm, body = vtb.split_frontmatter(text)
    res = vtb.check_h1_matches_frontmatter_title(body, fm)
    if (not res.passed) or res.is_warn:
        return res.detail
    return None


def _run_title_sync_sweep(tasks_root: Path | None = None) -> int:
    """Corpus-wide WARN-level H1-vs-frontmatter-title sync sweep (#1196).

    Iterates tasks/<status>/<N>/body.md (resolver-derived root — never a
    cwd-relative tasks/ path), printing one WARN row per flagged
    sentinelled body. ALWAYS returns 0: WARN only, never FAIL — whether
    the H1 or the frontmatter title is the fresher intent is a human
    call (each row's detail carries both values and both remediation
    commands, from the gate check). `tasks_root` is parameterized for
    tests; the production default is task_workflow.tasks_dir(). Read
    errors propagate (fail-fast, never swallowed).
    """
    if tasks_root is None:
        from explore_persona_space.task_workflow import tasks_dir

        tasks_root = tasks_dir()
    rows: list[tuple[int, str, str]] = []
    n_scanned = 0
    for body_path in sorted(tasks_root.glob("*/*/body.md")):
        tid = body_path.parent.name
        if not tid.isdigit():
            continue  # not a task folder (e.g. a non-numeric sibling dir)
        n_scanned += 1
        detail = h1_title_sync_warn(body_path.read_text(encoding="utf-8"))
        if detail:
            rows.append((int(tid), body_path.parent.parent.name, detail))
    print(f"Scanned {n_scanned} task bodies under {tasks_root}")
    if not rows:
        print("PASS: H1 == frontmatter title on every sentinelled clean-result body")
        return 0
    print(
        f"WARN: {len(rows)} sentinelled clean-result body(ies) with "
        "H1/frontmatter-title drift (WARN only — exit stays 0; which side is "
        "the fresher intent is a human call — each row's detail carries both "
        "values and both remediation commands)"
    )
    for tid, status, detail in sorted(rows):
        print(f"- #{tid} ({status}): {detail}")
    return 0


def _audit_single_body(body: str) -> int:
    """Audit one body: the `PASS:`/`FAIL:` headline + `- <name>: ...`
    findings rows (exit code per findings, exactly as before), then an
    advisory `WARN h1_title_sync:` line (#1196) when the H1 and the
    frontmatter title have drifted — WARN only, never touches the
    returned exit code."""
    if _is_paper_stub(body):
        print(
            "PASS: paper-task body.md is a paper-stub — markdown body-discipline "
            "checks skipped (the LaTeX paper is audited by verify_paper.py)"
        )
        return 0
    findings = audit_body(body)
    if not findings:
        print("PASS: no body-discipline anti-patterns matched")
        rc = 0
    else:
        print("FAIL: body-discipline anti-patterns matched")
        for name, samples in findings.items():
            print(f"- {name}: {', '.join(repr(s) for s in samples[:3])}")
        rc = 1
    warn = h1_title_sync_warn(body)
    if warn:
        print(f"WARN h1_title_sync: {warn}")
    return rc


def _run_legacy_bulk_inventory() -> None:
    """Legacy bulk-inventory mode: read pre-built inventory.json and write
    findings markdown across all awaiting-promotion items."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = list_awaiting_promotion()
    print(f"Found {len(items)} awaiting-promotion items")
    INVENTORY_PATH.write_text(json.dumps(items, indent=2))

    issue_findings: list[tuple[int, str, bool, dict[str, list[str]]]] = []
    for it in items:
        n = it["number"]
        body = gh("api", f"repos/superkaiba/explore-persona-space/issues/{n}", "--jq", ".body")
        v2 = is_v2(body)
        findings = audit_body(body) if v2 else {}
        issue_findings.append((n, it["title"], v2, findings))

    pattern_counts: dict[str, int] = {k: 0 for k in PATTERNS}
    issues_by_pattern: dict[str, list[int]] = {k: [] for k in PATTERNS}
    for n, _t, v2, findings in issue_findings:
        if not v2:
            continue
        for k in findings:
            pattern_counts[k] += 1
            issues_by_pattern[k].append(n)

    lines = ["# Body-discipline audit — 2026-05-08", ""]
    lines.append(f"Total awaiting-promotion items: {len(items)}")
    v2_count = sum(1 for _, _, v2, _ in issue_findings if v2)
    lines.append(f"v2-shape (migrated) items: {v2_count}")
    not_v2 = [(n, t) for n, t, v2, _ in issue_findings if not v2]
    lines.append(f"not v2-shape (unmigrated): {len(not_v2)}")
    lines.append("")

    lines.append("## Pattern frequency (across v2 items)")
    lines.append("")
    lines.append("| Pattern | Issues affected | Description |")
    lines.append("|---|---|---|")
    for k in sorted(pattern_counts, key=lambda k: -pattern_counts[k]):
        n_aff = pattern_counts[k]
        if n_aff == 0:
            continue
        ids = issues_by_pattern[k]
        ids_str = ", ".join(f"#{i}" for i in sorted(ids))
        lines.append(f"| `{k}` | {n_aff} ({ids_str}) | {PATTERNS[k][1]} |")
    lines.append("")

    lines.append("## Per-issue findings (v2 only)")
    lines.append("")
    for n, t, v2, findings in sorted(issue_findings):
        if not v2 or not findings:
            continue
        lines.append(f"### #{n} — {t[:80]}")
        for k, samples in findings.items():
            lines.append(
                f"- **{k}** ({len(samples)} sample(s)): {', '.join(repr(s) for s in samples[:3])}"
            )
        lines.append("")

    if not_v2:
        lines.append("## Not v2-shape (unmigrated, audit skipped)")
        lines.append("")
        for n, t in sorted(not_v2):
            lines.append(f"- #{n} — {t[:80]}")

    FINDINGS_PATH.write_text("\n".join(lines))
    print(f"Findings: {FINDINGS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit clean-result body prose for known discipline anti-patterns."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "body_file",
        nargs="?",
        help="Optional local markdown body to audit (e.g. an analyzer draft).",
    )
    src.add_argument(
        "--task",
        "--issue",
        dest="task",
        type=int,
        help="Task number; resolves to tasks/<status>/<N>/body.md. (--issue is an alias.)",
    )
    src.add_argument(
        "--title-sync-sweep",
        action="store_true",
        help="Corpus-wide WARN-level H1-vs-frontmatter-title sync sweep over "
        "tasks/*/*/body.md (#1196). One WARN row per divergent sentinelled "
        "clean-result body; WARN only — always exits 0.",
    )
    args = parser.parse_args()

    if args.title_sync_sweep:
        raise SystemExit(_run_title_sync_sweep())

    if args.task is not None:
        try:
            body_path = _resolve_task_body_path(args.task)
        except FileNotFoundError as exc:
            print(f"audit_clean_results_body_discipline: {exc}")
            raise SystemExit(2) from exc
        rc = _audit_single_body(body_path.read_text(encoding="utf-8"))
        if rc != 0:
            raise SystemExit(rc)
        return

    if args.body_file:
        rc = _audit_single_body(Path(args.body_file).read_text(encoding="utf-8"))
        if rc != 0:
            raise SystemExit(rc)
        return

    _run_legacy_bulk_inventory()


if __name__ == "__main__":
    main()
