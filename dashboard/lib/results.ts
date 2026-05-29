/**
 * Public clean-result catalog data layer (the /results surface).
 *
 * A "result" is a task that has been COMPLETED and promoted with the
 * AUTHORITATIVE `classification` frontmatter field set to "useful". The
 * predicate reads that structured field directly — it does NOT fall back to
 * the prose `isUsefulCleanResult` regex (which `lib/update-results.ts` used
 * for the legacy DB-backed feed). Tasks tagged `format-exemplar` are excluded
 * (they exist to demonstrate the body format, not to publish a finding).
 *
 * Everything here is server-side only (reads tasks/ from disk). The list/
 * detail routes select only public-safe fields before passing them to the
 * client; we never leak pod names, session ids, or in-flight body drafts.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { REGISTRY_PATH, type Status } from "./repo";
import { getRegistry, getTask, type Frontmatter } from "./tasks";

export type ResultConfidence = "HIGH" | "MODERATE" | "LOW" | null;

const LEGACY_SAGAN_CARD_SENTINEL = "<!-- legacy-sagan-card -->";
const FORMAT_EXEMPLAR_TAG = "format-exemplar";

/** A row in the public Results catalog (list view). Public-safe fields only. */
export type ResultListing = {
  id: number;
  /** Title with the `(… confidence)` tag stripped for display. */
  title: string;
  /** Raw title including the confidence tag (used for search). */
  rawTitle: string;
  confidence: ResultConfidence;
  tags: string[];
  excerpt: string;
  /** Promotion timestamp (ISO) when present, else body.md mtime (ISO). */
  date: string;
  /** YYYY-MM-DD form of `date`, for the date filter. */
  dayKey: string;
  href: string;
};

/** Full public result for the detail route. */
export type ResultDetail = {
  id: number;
  title: string;
  rawTitle: string;
  confidence: ResultConfidence;
  tags: string[];
  date: string;
  body: string;
  isLegacyHtml: boolean;
};

/**
 * Parse the trailing `(HIGH|MODERATE|LOW confidence)` tag the clean-result
 * spec requires on every title. Returns null when absent.
 */
export function parseConfidence(title: string): ResultConfidence {
  const m = title.match(/\((HIGH|MODERATE|LOW)\s+confidence\)\s*$/i);
  if (!m) return null;
  return m[1].toUpperCase() as ResultConfidence;
}

/** Strip the trailing `(… confidence)` tag for a cleaner card/heading. */
export function stripConfidenceTag(title: string): string {
  return title.replace(/\s*\((?:HIGH|MODERATE|LOW)\s+confidence\)\s*$/i, "").trim();
}

/**
 * The authoritative public predicate. A task is a public result iff:
 *   - status === "completed", AND
 *   - frontmatter.classification === "useful" (structured field, NOT a regex), AND
 *   - it is NOT tagged `format-exemplar`.
 *
 * `proposed` / `awaiting_promotion` / `archived` are excluded by the status
 * check (only `completed` passes).
 */
function isPublicResult(status: Status, fm: Frontmatter): boolean {
  if (status !== "completed") return false;
  const classification =
    typeof fm.classification === "string" ? fm.classification.trim() : "";
  if (classification !== "useful") return false;
  const tags = Array.isArray(fm.tags) ? (fm.tags as string[]) : [];
  if (tags.includes(FORMAT_EXEMPLAR_TAG)) return false;
  return true;
}

/** ~lightweight excerpt of a markdown body (mirrors update-results.markdownExcerpt). */
function markdownExcerpt(markdown: string, maxLength = 240): string {
  const plain = markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/^#.*$/m, " ") // drop the leading H1 title line
    .replace(/!\[[^\]]*]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/[#>*_`~|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (plain.length <= maxLength) return plain;
  const clipped = plain.slice(0, maxLength);
  const lastSpace = clipped.lastIndexOf(" ");
  return `${clipped.slice(0, lastSpace > 120 ? lastSpace : maxLength).trim()}...`;
}

function toDayKey(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/**
 * Resolve a task folder's body.md absolute path from a registry path entry.
 * Registry paths are repo-relative (e.g. "tasks/completed/390").
 */
function bodyPathForRegistryEntry(relPath: string): string {
  return path.join(path.dirname(REGISTRY_PATH), "..", relPath, "body.md");
}

/**
 * All public results, newest first. Reads the registry, then opens each
 * completed task's body.md frontmatter to apply the authoritative predicate.
 */
export function listPublicResults(): ResultListing[] {
  const reg = getRegistry();
  const out: ResultListing[] = [];

  for (const [idStr, entry] of Object.entries(reg.tasks)) {
    const id = Number(idStr);
    if (!Number.isFinite(id)) continue;
    if (entry.status !== "completed") continue; // cheap filter before disk read

    const bodyPath = bodyPathForRegistryEntry(entry.path);
    let raw: string;
    let stat: fs.Stats;
    try {
      stat = fs.statSync(bodyPath);
      raw = fs.readFileSync(bodyPath, "utf8");
    } catch {
      continue;
    }

    const fm = matter(raw);
    const data = fm.data as Frontmatter;
    if (!isPublicResult(entry.status as Status, data)) continue;

    const rawTitle =
      (typeof data.title === "string" && data.title.trim()) || entry.title || `Task #${id}`;
    const isLegacyHtml = fm.content.includes(LEGACY_SAGAN_CARD_SENTINEL);
    const bodyForExcerpt = isLegacyHtml ? "" : fm.content;
    const excerpt = bodyForExcerpt ? markdownExcerpt(bodyForExcerpt) : stripConfidenceTag(rawTitle);

    const promotedAt =
      typeof data.promoted_at === "string" ? Date.parse(data.promoted_at) : NaN;
    const dateMs = Number.isFinite(promotedAt) ? promotedAt : stat.mtimeMs;
    const date = new Date(dateMs);

    out.push({
      id,
      title: stripConfidenceTag(rawTitle),
      rawTitle,
      confidence: parseConfidence(rawTitle),
      tags: Array.isArray(data.tags) ? (data.tags as string[]) : [],
      excerpt,
      date: date.toISOString(),
      dayKey: toDayKey(date),
      href: `/results/${id}`,
    });
  }

  out.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : b.id - a.id));
  return out;
}

/**
 * Full public result by id, or null if the task does not exist OR fails the
 * public predicate (so a hand-crafted /results/<n> URL for a non-public task
 * 404s instead of leaking an in-flight body).
 */
export function getPublicResult(id: number): ResultDetail | null {
  if (!Number.isFinite(id)) return null;
  const task = getTask(id);
  if (!task) return null;
  if (!isPublicResult(task.status, task.frontmatter)) return null;

  const rawTitle =
    (typeof task.frontmatter.title === "string" && task.frontmatter.title.trim()) ||
    `Task #${id}`;
  const promotedAt =
    typeof task.frontmatter.promoted_at === "string"
      ? task.frontmatter.promoted_at
      : null;

  return {
    id,
    title: stripConfidenceTag(rawTitle),
    rawTitle,
    confidence: parseConfidence(rawTitle),
    tags: Array.isArray(task.frontmatter.tags) ? (task.frontmatter.tags as string[]) : [],
    date: promotedAt ?? "",
    body: task.body,
    isLegacyHtml: task.isLegacyHtml,
  };
}

/** Distinct sorted tag list across all public results (for the filter UI). */
export function publicResultTags(results: ResultListing[]): string[] {
  const set = new Set<string>();
  for (const r of results) for (const t of r.tags) set.add(t);
  return Array.from(set).sort((a, b) => a.localeCompare(b));
}
