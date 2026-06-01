/**
 * Server-side pre-render string transform: rewrite `#N` in evidence carrier
 * lines into clickable markdown links.
 *
 * Why server-side string substitution and not a React markdown plugin: the
 * shared <MarkdownDoc> client component cannot read the registry (it has no
 * disk access), so it must receive ALREADY-LINKIFIED markdown. Mirrors the
 * `rewriteMarkdownLinks` pattern in app/docs/[slug]/page.tsx.
 *
 * Scope (intentionally narrow — every guard below is load-bearing):
 *
 *   - Only blockquote lines (lines whose first non-whitespace char is `>`).
 *     Prose paragraphs, list bullets, headings, captions, tables — NEVER
 *     touched. App bullets at `## Applications` are list items, NOT
 *     blockquotes, so their inline `#N` examples / dependencies stay raw.
 *   - The blockquote line must contain `**Evidence:**` (Belief carrier) OR
 *     match the State trailer header `**State:** ... evidence:`. A
 *     blockquote line that is JUST a `*Next:*` rider or a stray quote is
 *     left untouched, because its `#N` aren't part of an evidence list
 *     (q3.9 has `(#445)` inside a `*Next:*` line we deliberately don't link
 *     — the surrounding parenthetical there is prose).
 *   - Code spans (`` `#123` ``) and math (`$ ... $`) inside a carrier line
 *     are left untouched: we mask them before substitution and restore
 *     after.
 *   - The substitution starts at the first character of the evidence VALUE
 *     (i.e. after `**Evidence:**` or `evidence:` on a State line), so a
 *     `#N` accidentally sitting in the Belief prose before `**Evidence:**`
 *     (the wider blockquote may carry inline citations) is NOT linkified —
 *     only the structured evidence list itself.
 */
import { evidenceHrefForTaskId } from "./results";
import { isEmptyEvidenceValue } from "./questions";

const BLOCKQUOTE_LINE = /^(\s*>\s.*)$/;
const BELIEF_EVIDENCE_MARKER = "**Evidence:**";
const STATE_EVIDENCE_MARKER = /\bevidence:\s*/i;
const TASK_REF = /#(\d+)/g;

// Token mask helpers — protect code spans + inline math so a stray `#N`
// inside `` `#123` `` or `$#123$` is left raw.
type Masked = { text: string; restore: (s: string) => string };

function maskInline(line: string): Masked {
  const codeStash: string[] = [];
  const mathStash: string[] = [];
  const masked = line
    .replace(/`[^`]*`/g, (m) => {
      codeStash.push(m);
      return `C${codeStash.length - 1}`;
    })
    .replace(/\$[^$\n]+\$/g, (m) => {
      mathStash.push(m);
      return `M${mathStash.length - 1}`;
    });
  return {
    text: masked,
    restore: (s) =>
      s
        .replace(/C(\d+)/g, (_m, i) => codeStash[Number(i)])
        .replace(/M(\d+)/g, (_m, i) => mathStash[Number(i)]),
  };
}

/**
 * Substitute `#N` -> `[#N](href)` in `evidenceSegment` only. The href is
 * decided per-id by `evidenceHrefForTaskId(id, publicIds)`.
 */
function linkifyEvidenceSegment(
  evidenceSegment: string,
  publicIds: Set<number>,
): string {
  // Empty-evidence sentinel (`none in-house yet`, optionally with a
  // parenthetical `#N` aside) is NOT a list — leave it untouched so the
  // render agrees with `parseEvidence` (which returns [] for it). Without
  // this, `none in-house yet (... #428)` would linkify #428 while the
  // /questions hub shows no evidence row for it.
  if (isEmptyEvidenceValue(evidenceSegment)) return evidenceSegment;
  const masked = maskInline(evidenceSegment);
  const out = masked.text.replace(TASK_REF, (whole, idStr) => {
    const id = Number(idStr);
    if (!Number.isFinite(id)) return whole;
    return `[#${id}](${evidenceHrefForTaskId(id, publicIds)})`;
  });
  return masked.restore(out);
}

/**
 * Rewrite all evidence `#N` in carrier blockquote lines into markdown links.
 *
 * Iterates the doc line-by-line. For each blockquote line:
 *   1. If it contains `**Evidence:**` (Belief-format carrier), substitute
 *      everything AFTER the marker.
 *   2. Else if it matches `**State:** ... evidence:` (State-trailer
 *      carrier), substitute everything AFTER `evidence:`.
 *   3. Else leave the line alone.
 *
 * Non-blockquote lines are passed through unchanged, so app bullets,
 * captions, headings, and prose retain raw `#N` references.
 */
export function linkifyEvidenceInOpenQuestions(
  markdown: string,
  publicIds: Set<number>,
): string {
  const lines = markdown.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!BLOCKQUOTE_LINE.test(line)) continue;

    const evIdx = line.indexOf(BELIEF_EVIDENCE_MARKER);
    if (evIdx !== -1) {
      const head = line.slice(0, evIdx + BELIEF_EVIDENCE_MARKER.length);
      const tail = line.slice(evIdx + BELIEF_EVIDENCE_MARKER.length);
      lines[i] = head + linkifyEvidenceSegment(tail, publicIds);
      continue;
    }
    // Only treat `evidence:` as a marker when it's in a State trailer line
    // — i.e. the line begins with `> **State:** ...`. Otherwise a stray
    // lowercase `evidence:` in prose would be eaten.
    if (/^\s*>\s*\*\*State:\*\*/.test(line)) {
      const m = STATE_EVIDENCE_MARKER.exec(line);
      if (m) {
        const cut = m.index + m[0].length;
        const head = line.slice(0, cut);
        const tail = line.slice(cut);
        lines[i] = head + linkifyEvidenceSegment(tail, publicIds);
      }
    }
  }
  return lines.join("\n");
}
