/**
 * Shared markdown heading extraction + slug helpers.
 *
 * Used by:
 *   - InteractiveResult.tsx (small-card heading rendering)
 *   - TocSidebar.tsx (modal sidebar)
 *   - CommentableBody.tsx (heading id + collapsible wrap)
 *
 * Slugs follow GitHub's rules approximately: lowercase, strip punctuation,
 * spaces → hyphens, collisions get `-2`, `-3`, ...
 */

export type MarkdownHeading = {
  id: string;
  text: string;
  depth: number;
  /** Order within the document (0-indexed). */
  index: number;
};

const HEADING_PATTERN = /^(#{1,6})\s+(.+?)\s*#*\s*$/gm;

/**
 * Parse the markdown source and return one entry per heading (depth 1-6).
 *
 * The function ignores headings inside fenced code blocks ```...```.
 */
export function extractMarkdownHeadings(markdown: string): MarkdownHeading[] {
  const stripped = stripFencedCodeBlocks(markdown);
  const counts = new Map<string, number>();
  const headings: MarkdownHeading[] = [];
  const pattern = new RegExp(HEADING_PATTERN.source, "gm");
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(stripped))) {
    const text = plainMarkdownText(match[2]);
    if (!text) continue;
    const baseId = githubLikeSlug(text);
    const id = dedupeSlug(baseId, counts);
    headings.push({
      id,
      text,
      depth: match[1].length,
      index: headings.length,
    });
  }

  return headings;
}

/** Strip inline markdown formatting so the heading text is plain. */
export function plainMarkdownText(value: string): string {
  return value
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[([^\]]*)]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/[*_~]/g, "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function githubLikeSlug(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, "")
    .replace(/\s+/g, "-");
  return slug || "section";
}

export function dedupeSlug(baseId: string, counts: Map<string, number>): string {
  const count = counts.get(baseId) ?? 0;
  counts.set(baseId, count + 1);
  return count === 0 ? baseId : `${baseId}-${count}`;
}

/**
 * Replace fenced code blocks (``` ... ```) with the same number of blank
 * lines so headings inside code don't get picked up but line offsets are
 * preserved if any caller cares.
 */
function stripFencedCodeBlocks(markdown: string): string {
  return markdown.replace(/```[\s\S]*?```/g, (block) => {
    const newlines = block.match(/\n/g)?.length ?? 0;
    return "\n".repeat(newlines);
  });
}
