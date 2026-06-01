/**
 * Research-hub questions parser: `docs/open_questions.md` -> typed `Question[]`.
 *
 * Reads the SAME doc that `scripts/living_docs.py` writes / lints. The Python
 * module is the canonical write-side (apply / link / check); this is the
 * read-side for the dashboard. Both parsers must agree on the carrier-line
 * grammar so a dashboard render of the doc lines up with what `living_docs.py`
 * understands. The carrier regexes + section-boundary rule + empty-evidence
 * sentinel set are ported faithfully (see references at each regex).
 *
 * Section grouping is by HEADING LEVEL (NOT a naive "nearest H3"):
 *
 *   - `## Applications`  H2 region -> every anchor in it is an `application`
 *     (free-text **Status:** in the bullet, NO evidence carrier, contributes
 *     ZERO edges to the reverse index).
 *   - `## Settled`       H2 region -> question with `status: "settled"`.
 *   - `## Open questions` H2 region -> a question's `section` is the nearest
 *     preceding `### N. <name>` H3 in the SAME region; a `subsection` is the
 *     nearest preceding H4 (if any).
 *   - Anchors under Motivation / Framing / Glossary (or any other H2 region)
 *     are not questions and are skipped.
 *
 * Heading state is reset on each `## ...` H2 transition AND on each `---`
 * hrule. The changelog block (`<!-- living-docs-changelog:begin/end -->`)
 * is injected by `apply()` on first run; we tolerate it being absent today
 * and SKIP its contents when present (no anchor + no `#N` parsing inside it).
 *
 * Evidence is extracted ONLY from the per-question carrier line — either a
 *   > **State:** ... evidence: #N, #M
 * (canonical schema, what new stubs emit) OR a
 *   > **Belief:** ... **Confidence:** ... **Evidence:** #N, #M
 * (live form used by every question in the current doc; **Confidence:** and
 * **Evidence:** may sit on a LATER blockquote line than **Belief:** — e.g.
 * q3.1's three-line trailer with a `*Next:*` line between Belief and Evidence).
 * Evidence is NEVER scraped from question titles (e.g. q3.4 title contains
 * `#383`), from inline prose, or from app bullets.
 *
 * Public-vs-gated link routing lives in `lib/results.ts` so the dashboard has
 * ONE source of truth for the `classification:useful` predicate (the overview
 * transform, /questions, and /results/[id] all import the same helper).
 */
import fs from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./repo";

const OPEN_QUESTIONS_PATH = path.join(REPO_ROOT, "docs", "open_questions.md");

const CHANGELOG_BEGIN = "<!-- living-docs-changelog:begin -->";
const CHANGELOG_END = "<!-- living-docs-changelog:end -->";

// Anchor: ``<!-- q:<id> -->``. Mirrors `_ANCHOR_RE` in living_docs.py
// (case-insensitive ids; we lowercase on read).
const ANCHOR_RE = /<!--\s*q:([A-Za-z0-9_.\-]+)\s*-->/;

// State trailer carrier. Mirrors `_STATE_RE`.
//   > **State:** 🌿 budding · MODERATE · updated 2026-05-28 · evidence: #207, #380
const STATE_RE =
  /^>\s*\*\*State:\*\*\s*\S+?(?:\s+[^·]*?)?\s*·\s*(LOW|MODERATE|HIGH)\s*·\s*updated\s+\d{4}-\d{2}-\d{2}\s*·\s*evidence:\s*(.*?)\s*$/;

// Belief-format Evidence carrier line. Mirrors `_BELIEF_EVIDENCE_RE`. The
// blockquote line may carry **Belief:** + **Confidence:** + **Evidence:** all
// together (most common), or only **Evidence:** when the **Belief:** /
// **Confidence:** segments sit on EARLIER blockquote lines in the same
// section. The regex captures the evidence VALUE only.
const BELIEF_EVIDENCE_RE = /^>.*?\*\*Evidence:\*\*\s+(.*?)\.?\s*$/;

// Standalone Belief / Confidence carrier lines (when split across multiple
// blockquote lines in the same section). The evidence carrier above handles
// the row that includes **Evidence:**; these two extractors recover the
// `belief` and `confidence` fields when those segments live on earlier rows.
const BELIEF_PROSE_RE = /\*\*Belief:\*\*\s+(.*?)(?:\s*\*\*Confidence:\*\*|\s*$)/;
const CONFIDENCE_RE = /\*\*Confidence:\*\*\s+(LOW|MODERATE|HIGH)/;

// Optional `> *Next: ...*` blockquote line for a question.
const NEXT_RE = /^>\s*\*Next:\s*(.+?)\*\s*$/;

// A `#NNN` task reference. Used by evidence-list parsing AND by the
// downstream linkify transform.
const TASK_REF_RE = /#(\d+)/g;

// Empty-evidence sentinels. Mirrors `_EMPTY_BELIEF_VALUES`. Gating on this
// set (not "no #N in the value") is intentional — a value like
//   "none in-house yet (definitional groundwork tracked in #428)"
// carries `#428` inside a parenthetical aside, which must NOT be parsed as
// evidence (the parenthetical is prose, not a list).
const EMPTY_BELIEF_VALUES = new Set([
  "none in-house yet",
  "none yet",
  "tbd",
  "none",
]);

/**
 * True when an evidence VALUE is an empty-evidence sentinel — either bare
 * (`none in-house yet`) or a sentinel followed by a parenthetical aside
 * (`none in-house yet (definitional groundwork tracked in #428)`). In the
 * latter case the `#428` is prose, NOT a list entry.
 *
 * Shared by `parseEvidence` (so the question's evidence is []) AND by the
 * server-side linkify transform (so the `#428` is NOT turned into a link) —
 * the two surfaces MUST agree on what counts as "no evidence", or the doc
 * render links an id the /questions hub doesn't list.
 */
export function isEmptyEvidenceValue(value: string): boolean {
  const trimmed = value.trim().replace(/\.\s*$/, "").toLowerCase();
  if (EMPTY_BELIEF_VALUES.has(trimmed)) return true;
  for (const sentinel of EMPTY_BELIEF_VALUES) {
    if (trimmed.startsWith(`${sentinel} (`)) return true;
  }
  return false;
}

// App bullet: `- **App N — <gloss>** ... **Status: <free text>.** ...`
// The bullet ends at the next bullet OR the next `---` hrule. The bullet
// carries an anchor `<!-- q:appN -->` somewhere in its body.
const APP_BULLET_HEAD_RE = /^-\s+\*\*App\s+(\d+)\s+[—–-]\s+(.+?)\*\*/;
const APP_STATUS_RE = /\*\*Status:\s+(.+?)\.\*\*/;

// Question-section H3: `### N. <name>` or `### N <name>` under `## Open questions`.
const SECTION_H3_RE = /^###\s+(\d+)\.\s+(.+?)\s*$/;
const SUBSECTION_H4_RE = /^####\s+(.+?)\s*$/;

// Question heading shape, e.g. `**3.4a How do contrastive negatives shape leakage?**`
// or `**1.1 Can a context be treated as a vector?**`. The numeric prefix is
// `N.N` or `N.Na`-style; we strip it for the display title.
const QUESTION_HEADING_RE = /^\*\*([0-9]+(?:\.[0-9]+[a-z]?)?)\s+(.+?)\*\*/;

export type QuestionKind = "question" | "application";
export type QuestionStatus = "open" | "settled";
export type QuestionConfidence = "LOW" | "MODERATE" | "HIGH";

export type Question = {
  /** Anchor id, e.g. `spec-context-as-vector` or `app1`. Lowercase. */
  id: string;
  /**
   * Display-only number string. For questions: the `N.N`/`N.Na` prefix from
   * the heading (e.g. `3.4a`, `1.2`). For applications: the app index from
   * the bullet (e.g. `1`, `6`). May be empty when the heading lacks a number
   * (only the Motivation / Framing prose blocks lack it, and those carry no
   * anchors — so empty is purely a safety default).
   */
  number: string;
  /** Title with `**` stripped and the leading number stripped. */
  title: string;
  /**
   * Section bucket. For questions: the section H3 name (`Distance between
   * contexts`, `Updating (W, C) toward a behavior — what installs, at what
   * cost?`, `Generalization — how an update at (C, B) propagates to (C′, B′)`,
   * or `What are contexts and behaviors — the C–B duality`). For
   * applications: `"Applications"`. For settled questions: `"Settled"`.
   */
  section: string;
  /** Optional H4 subsection name (e.g. `Persona leakage (same behavior, a new persona)`). */
  subsection?: string;
  kind: QuestionKind;
  status: QuestionStatus;
  /** Question confidence; null for applications and questions with no carrier. */
  confidence: QuestionConfidence | null;
  /** One-sentence belief; null when absent (applications). */
  belief: string | null;
  /** Optional `*Next: ...*` text (without the `*Next:` / `*` markers). */
  next?: string;
  /**
   * Sorted, deduplicated, non-empty `#N` ids from the carrier line. EMPTY for
   * applications (apps carry only free-text status + inline example refs that
   * are prose, NOT evidence) and for questions whose carrier value is the
   * empty-evidence sentinel.
   */
  evidence: number[];
  /** Free-text app status, when `kind === "application"`. */
  appStatus?: string;
};

/* -------------------------------------------------------------------------- *
 * Parser
 * -------------------------------------------------------------------------- */

/** Strip the changelog block (when present). Tolerates its absence. */
function stripChangelog(text: string): string {
  const begin = text.indexOf(CHANGELOG_BEGIN);
  const end = text.indexOf(CHANGELOG_END);
  if (begin === -1 || end === -1 || end < begin) return text;
  return text.slice(0, begin) + text.slice(end + CHANGELOG_END.length);
}

/** Parse the evidence VALUE string into deduplicated, sorted task ids. */
function parseEvidence(value: string): number[] {
  // Empty-evidence sentinel (bare or with a parenthetical `#N` aside) → no
  // evidence. Shared with the linkify transform via `isEmptyEvidenceValue`.
  if (isEmptyEvidenceValue(value)) return [];
  const ids = new Set<number>();
  for (const m of value.matchAll(TASK_REF_RE)) {
    const n = Number(m[1]);
    if (Number.isFinite(n)) ids.add(n);
  }
  return Array.from(ids).sort((a, b) => a - b);
}

type ParseState = {
  /**
   * Current H2 region: "open" (under `## Open questions`), "applications"
   * (under `## Applications`), "settled" (under `## Settled`), or "other"
   * (Motivation / Framing / Glossary / before-any-H2 — anchors here are
   * skipped).
   */
  region: "open" | "applications" | "settled" | "other";
  /** Current H3 name within the open-questions region. Reset on H2. */
  section: string | null;
  /** Current H4 name. Reset on H2 OR H3 OR `---` hrule. */
  subsection: string | null;
  /** True while inside the changelog block (defense-in-depth; we also strip). */
  inChangelog: boolean;
};

function newState(): ParseState {
  return { region: "other", section: null, subsection: null, inChangelog: false };
}

/**
 * Find the carrier line + accompanying Belief/Confidence/Next for a question
 * heading at line `headingIdx`. Search forward until the next anchor, the next
 * `---` hrule, OR the next H2/H3/H4. Returns the parsed fields when a carrier
 * is found, else `null` (no carrier = question is rendered without
 * confidence/evidence/belief).
 */
function findQuestionTrailer(
  lines: string[],
  headingIdx: number,
): {
  belief: string | null;
  confidence: QuestionConfidence | null;
  evidence: number[];
  next?: string;
} | null {
  let belief: string | null = null;
  let confidence: QuestionConfidence | null = null;
  let evidence: number[] | null = null;
  let next: string | undefined;
  for (let i = headingIdx + 1; i < lines.length; i++) {
    const line = lines[i];
    if (i !== headingIdx && ANCHOR_RE.test(line)) break;
    if (line.trim() === "---") break;
    if (/^#{1,4}\s/.test(line)) break; // any new heading ends the section
    // Carrier line — State trailer wins if both happened to exist (mirror
    // living_docs.py preference). Otherwise Belief-format Evidence line.
    const stateMatch = STATE_RE.exec(line);
    if (stateMatch) {
      confidence = stateMatch[1] as QuestionConfidence;
      evidence = parseEvidence(stateMatch[2]);
      break;
    }
    const beliefEvidenceMatch = BELIEF_EVIDENCE_RE.exec(line);
    if (beliefEvidenceMatch) {
      // Recover **Belief:** / **Confidence:** from THIS row if present.
      if (belief === null) {
        const bp = BELIEF_PROSE_RE.exec(line);
        if (bp) belief = bp[1].trim().replace(/\s*\*\*$/, "").trim();
      }
      if (confidence === null) {
        const cm = CONFIDENCE_RE.exec(line);
        if (cm) confidence = cm[1] as QuestionConfidence;
      }
      evidence = parseEvidence(beliefEvidenceMatch[1]);
      // The Evidence carrier closes the trailer search; **Next:** can sit
      // just below it though, so keep scanning a few lines.
      for (let j = i + 1; j < lines.length; j++) {
        const after = lines[j];
        if (after.trim() === "---") break;
        if (/^#{1,4}\s/.test(after)) break;
        if (ANCHOR_RE.test(after)) break;
        const nx = NEXT_RE.exec(after);
        if (nx) {
          next = nx[1].trim();
          break;
        }
        if (!after.trim().startsWith(">")) break;
      }
      break;
    }
    // Non-carrier blockquote rows may carry **Belief:** or **Confidence:** in
    // isolation, or a **Next:** rider that sits BEFORE the **Evidence:** row
    // (q3.1's three-line trailer is the canonical case).
    if (line.trim().startsWith(">")) {
      const bp = BELIEF_PROSE_RE.exec(line);
      if (bp && belief === null) belief = bp[1].trim().replace(/\s*\*\*$/, "").trim();
      const cm = CONFIDENCE_RE.exec(line);
      if (cm && confidence === null) confidence = cm[1] as QuestionConfidence;
      const nx = NEXT_RE.exec(line);
      if (nx && next === undefined) next = nx[1].trim();
    }
  }
  if (evidence === null && belief === null && confidence === null) return null;
  return { belief, confidence, evidence: evidence ?? [], next };
}

/**
 * Parse an application bullet that starts at `bulletStart`. Returns the
 * collected `Question` (kind=application), or null when the bullet doesn't
 * carry a `<!-- q:appN -->` anchor.
 *
 * Bullets are multi-line and end at the next list bullet at the SAME
 * indentation level, the next `---` hrule, or the next heading.
 */
function parseApplicationBullet(
  lines: string[],
  bulletStart: number,
): { question: Question | null; consumedUpTo: number } {
  const headMatch = APP_BULLET_HEAD_RE.exec(lines[bulletStart]);
  if (!headMatch) return { question: null, consumedUpTo: bulletStart };

  const number = headMatch[1];
  // Drop trailing parenthetical gloss from the title for compactness.
  const rawTitle = headMatch[2].trim();

  // Collect the full bullet body.
  const parts: string[] = [lines[bulletStart]];
  let i = bulletStart + 1;
  for (; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === "---") break;
    if (/^#{1,4}\s/.test(line)) break;
    if (/^-\s+\*\*App\s+\d+/.test(line)) break;
    // A blank line followed by a non-indented bullet ends the app block.
    if (line === "" && i + 1 < lines.length && /^-\s+/.test(lines[i + 1])) break;
    parts.push(line);
  }
  const body = parts.join("\n");

  const anchorMatch = ANCHOR_RE.exec(body);
  if (!anchorMatch) return { question: null, consumedUpTo: i };
  const id = anchorMatch[1].toLowerCase();

  const statusMatch = APP_STATUS_RE.exec(body);
  const appStatus = statusMatch ? statusMatch[1].trim() : "unknown";

  return {
    question: {
      id,
      number,
      title: rawTitle,
      section: "Applications",
      kind: "application",
      status: "open", // apps don't graduate; "open" is the structural default
      confidence: null,
      belief: null,
      // Apps NEVER contribute evidence edges. The `#N` inside their prose are
      // examples / dependencies, not a structured evidence list.
      evidence: [],
      appStatus,
    },
    consumedUpTo: i,
  };
}

/**
 * Parse `docs/open_questions.md` into a Question[].
 *
 * Pure function over a string — `parseQuestionsFromMarkdown(...)` is the test
 * seam (the fixture corpus runs against it directly).
 */
export function parseQuestionsFromMarkdown(text: string): Question[] {
  const cleaned = stripChangelog(text);
  const lines = cleaned.split("\n");
  const state = newState();
  const out: Question[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // H2 transition resets H3/H4 and re-classifies the region.
    if (line.startsWith("## ") && !line.startsWith("### ")) {
      const h2 = line.slice(3).trim().toLowerCase();
      if (h2 === "open questions") state.region = "open";
      else if (h2 === "applications") state.region = "applications";
      else if (h2 === "settled") state.region = "settled";
      else state.region = "other";
      state.section = null;
      state.subsection = null;
      continue;
    }
    // `---` hrule resets H3/H4 (within a region; the H2 stays).
    if (line.trim() === "---") {
      state.section = null;
      state.subsection = null;
      continue;
    }
    // H3 inside the open-questions region names the current `### N. <name>`.
    if (state.region === "open") {
      const h3 = SECTION_H3_RE.exec(line);
      if (h3) {
        state.section = h3[2].trim();
        state.subsection = null;
        continue;
      }
    }
    // Any H3 outside the open-questions region (or non-numbered H3) — just
    // reset the H4 cursor so we don't carry a stale subsection across.
    if (line.startsWith("### ")) {
      state.subsection = null;
      continue;
    }
    // H4 = subsection within the current H3.
    const h4 = SUBSECTION_H4_RE.exec(line);
    if (h4) {
      state.subsection = h4[1].trim();
      continue;
    }

    // Applications: walk app bullets.
    if (state.region === "applications") {
      if (line.startsWith("- **App ")) {
        const { question, consumedUpTo } = parseApplicationBullet(lines, i);
        if (question) out.push(question);
        i = consumedUpTo - 1; // for-loop will ++ to consumedUpTo
        continue;
      }
      // Non-bullet anchors in Applications: skip (none today; orientation prose).
      continue;
    }

    // Open / Settled: look for a question heading line carrying an anchor.
    if (state.region === "open" || state.region === "settled") {
      const anchorMatch = ANCHOR_RE.exec(line);
      if (!anchorMatch) continue;
      const id = anchorMatch[1].toLowerCase();

      // Extract the question heading on the same line. The doc style is
      //   `**N.Na <title>** <!-- q:<id> -->`
      // i.e. the bolded heading + the anchor on one line. If we can't find a
      // numbered heading, fall back to whatever text precedes the anchor —
      // this keeps the parser total for any future stub / orientation anchor.
      const headingMatch = QUESTION_HEADING_RE.exec(line);
      let number = "";
      let title = "";
      if (headingMatch) {
        number = headingMatch[1];
        title = headingMatch[2].trim();
      } else {
        // Strip the anchor + bold markers; whatever remains is the title.
        title = line
          .replace(ANCHOR_RE, "")
          .replace(/\*\*/g, "")
          .trim();
      }
      // Section must be set for an open-region anchor; if it isn't, we're
      // under Motivation/Framing/Glossary or before the first `### N.` — skip
      // (none of those carry question anchors today).
      if (state.region === "open" && !state.section) continue;

      const trailer = findQuestionTrailer(lines, i);

      out.push({
        id,
        number,
        title,
        section: state.region === "settled" ? "Settled" : (state.section as string),
        subsection: state.subsection ?? undefined,
        kind: "question",
        status: state.region === "settled" ? "settled" : "open",
        confidence: trailer?.confidence ?? null,
        belief: trailer?.belief ?? null,
        next: trailer?.next,
        evidence: trailer?.evidence ?? [],
      });
    }
  }

  return out;
}

/* -------------------------------------------------------------------------- *
 * Disk-backed entry points
 * -------------------------------------------------------------------------- */

let _cached: { mtimeMs: number; questions: Question[] } | null = null;

/**
 * Read + parse `docs/open_questions.md` from the repo. Caches on body.md mtime
 * so a single request can call this multiple times (e.g. the /questions page
 * + the reverse-index helpers) without re-reading.
 */
export function listQuestions(): Question[] {
  try {
    const stat = fs.statSync(OPEN_QUESTIONS_PATH);
    if (_cached && _cached.mtimeMs === stat.mtimeMs) return _cached.questions;
    const raw = fs.readFileSync(OPEN_QUESTIONS_PATH, "utf8");
    const questions = parseQuestionsFromMarkdown(raw);
    _cached = { mtimeMs: stat.mtimeMs, questions };
    return questions;
  } catch {
    return [];
  }
}

/** Reverse index: which questions list `taskId` as evidence. Apps contribute none. */
export function questionsForResult(taskId: number): Question[] {
  if (!Number.isFinite(taskId)) return [];
  return listQuestions().filter(
    (q) => q.kind !== "application" && q.evidence.includes(taskId),
  );
}

/** Path to the source markdown (exported for tests / debugging only). */
export const OPEN_QUESTIONS_MARKDOWN_PATH = OPEN_QUESTIONS_PATH;
