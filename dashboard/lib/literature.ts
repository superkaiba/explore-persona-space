/**
 * Read updates/literature/{YYYY-MM-DD.md, papers/<slug>.md} from disk.
 *
 * All functions are server-side only.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { REPO_ROOT } from "./repo";

export const LITERATURE_DIR = path.join(REPO_ROOT, "updates", "literature");
export const PAPERS_DIR = path.join(LITERATURE_DIR, "papers");

const DATE_FILE_RE = /^\d{4}-\d{2}-\d{2}\.md$/;

export type DailyBatchFrontmatter = {
  date?: string;
  item_count?: number;
  top_score?: number;
  generated_at?: string;
  [k: string]: unknown;
};

export type DailyBatchListing = {
  date: string;
  itemCount: number;
  topScore: number;
};

export type DailyBatch = {
  date: string;
  frontmatter: DailyBatchFrontmatter;
  body: string;
};

export type PaperFrontmatter = {
  arxiv_id?: string | null;
  lit_item_id?: string;
  title?: string;
  authors?: string[];
  topic?: string | null;
  released_on?: string | null;
  url?: string | null;
  pdf_url?: string | null;
  first_surfaced_on?: string;
  highest_score?: number;
  categories?: string[];
  surfaced_days?: string[];
  tags?: string[];
  [k: string]: unknown;
};

export type Paper = {
  slug: string;
  frontmatter: PaperFrontmatter;
  body: string;
};

function safeReadDir(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir);
}

export function listDailyBatches(): DailyBatchListing[] {
  const entries = safeReadDir(LITERATURE_DIR)
    .filter((name) => DATE_FILE_RE.test(name))
    .sort()
    .reverse();
  const rows: DailyBatchListing[] = [];
  for (const name of entries) {
    const full = path.join(LITERATURE_DIR, name);
    const raw = fs.readFileSync(full, "utf-8");
    const { data } = matter(raw);
    const fm = data as DailyBatchFrontmatter;
    rows.push({
      date: fm.date ?? name.replace(/\.md$/, ""),
      itemCount: typeof fm.item_count === "number" ? fm.item_count : 0,
      topScore: typeof fm.top_score === "number" ? fm.top_score : 0,
    });
  }
  return rows;
}

export function getDailyBatch(date: string): DailyBatch | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  const full = path.join(LITERATURE_DIR, `${date}.md`);
  if (!fs.existsSync(full)) return null;
  const raw = fs.readFileSync(full, "utf-8");
  const parsed = matter(raw);
  return {
    date,
    frontmatter: parsed.data as DailyBatchFrontmatter,
    body: parsed.content,
  };
}

export function getPaper(slug: string): Paper | null {
  // Slugs are arxiv ids (digits + optional dot/letters) or 12-char hex.
  // Reject anything with a path separator to keep us inside PAPERS_DIR.
  if (!/^[A-Za-z0-9._-]+$/.test(slug)) return null;
  const full = path.join(PAPERS_DIR, `${slug}.md`);
  if (!fs.existsSync(full)) return null;
  const raw = fs.readFileSync(full, "utf-8");
  const parsed = matter(raw);
  return {
    slug,
    frontmatter: parsed.data as PaperFrontmatter,
    body: parsed.content,
  };
}
