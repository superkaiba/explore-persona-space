/**
 * Read tasks/<status>/<id>/{body.md, events.jsonl, comments.jsonl} from disk.
 *
 * All functions are server-side only.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { REGISTRY_PATH, STATUSES, type Status } from "./repo";

export type Registry = {
  highest_id: number;
  tasks: Record<
    string,
    {
      path: string;
      title: string;
      kind: string;
      status: string;
      has_clean_result: boolean;
    }
  >;
};

export type Frontmatter = {
  title?: string;
  kind?: string;
  tags?: string[];
  created_at?: string;
  parent_id?: number;
  pod_name?: string;
  happy_session_id?: string;
  has_clean_result?: boolean;
  classification?: string;
  promoted_at?: string;
  sagan_id?: string;
  sagan_number?: number;
  priority?: string;
  [k: string]: unknown;
};

export type Task = {
  id: number;
  status: Status;
  path: string; // absolute
  frontmatter: Frontmatter;
  body: string;
  isLegacyHtml: boolean;
};

export type TaskListing = {
  id: number;
  title: string;
  kind: string;
  status: Status;
  tags: string[];
  hasCleanResult: boolean;
  classification?: string;
};

export type TaskEvent = {
  ts: string;
  kind: string;
  version?: number;
  by?: string;
  note?: string;
  from?: string;
  to?: string;
  [k: string]: unknown;
};

export type TaskComment = {
  id: string;
  ts: string;
  author: string;
  kind: "question" | "answer" | "followup-proposal" | "note";
  body: string;
  in_reply_to?: string;
  [k: string]: unknown;
};

const LEGACY_SAGAN_CARD_SENTINEL = "<!-- legacy-sagan-card -->";

function readRegistryRaw(): Registry {
  const raw = fs.readFileSync(REGISTRY_PATH, "utf8");
  return JSON.parse(raw) as Registry;
}

export function getRegistry(): Registry {
  try {
    return readRegistryRaw();
  } catch {
    return { highest_id: 0, tasks: {} };
  }
}

export function resolveTaskPath(id: number): string | null {
  const reg = getRegistry();
  const entry = reg.tasks[String(id)];
  if (entry) {
    const abs = path.join(path.dirname(REGISTRY_PATH), "..", entry.path);
    if (fs.existsSync(abs)) return abs;
  }
  // Fallback scan
  for (const status of STATUSES) {
    const candidate = path.join(path.dirname(REGISTRY_PATH), status, String(id));
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

export function getTask(id: number): Task | null {
  const abs = resolveTaskPath(id);
  if (!abs) return null;
  const bodyPath = path.join(abs, "body.md");
  if (!fs.existsSync(bodyPath)) return null;
  const raw = fs.readFileSync(bodyPath, "utf8");
  // Strip legacy sentinel BEFORE gray-matter parses, since the sentinel
  // sits between the frontmatter close (---) and the HTML body — gray-matter
  // already correctly leaves it in body, so we just record the flag.
  const fm = matter(raw);
  const status = path.basename(path.dirname(abs)) as Status;
  const body = fm.content;
  const isLegacyHtml = body.includes(LEGACY_SAGAN_CARD_SENTINEL);
  return {
    id,
    status,
    path: abs,
    frontmatter: fm.data as Frontmatter,
    body: isLegacyHtml ? body.replace(LEGACY_SAGAN_CARD_SENTINEL, "").trimStart() : body,
    isLegacyHtml,
  };
}

export function getEvents(id: number): TaskEvent[] {
  const abs = resolveTaskPath(id);
  if (!abs) return [];
  const p = path.join(abs, "events.jsonl");
  if (!fs.existsSync(p)) return [];
  const raw = fs.readFileSync(p, "utf8");
  return raw
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as TaskEvent);
}

export function getComments(id: number): TaskComment[] {
  const abs = resolveTaskPath(id);
  if (!abs) return [];
  const p = path.join(abs, "comments.jsonl");
  if (!fs.existsSync(p)) return [];
  const raw = fs.readFileSync(p, "utf8");
  return raw
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as TaskComment);
}

export function listAllTasks(): TaskListing[] {
  const reg = getRegistry();
  const out: TaskListing[] = [];
  for (const [idStr, entry] of Object.entries(reg.tasks)) {
    const id = Number(idStr);
    if (!Number.isFinite(id)) continue;
    if (!STATUSES.includes(entry.status as Status)) continue;
    // Read frontmatter for richer fields
    let tags: string[] = [];
    let classification: string | undefined;
    try {
      const abs = path.join(path.dirname(REGISTRY_PATH), "..", entry.path);
      const raw = fs.readFileSync(path.join(abs, "body.md"), "utf8");
      const fm = matter(raw).data as Frontmatter;
      tags = Array.isArray(fm.tags) ? fm.tags : [];
      classification = typeof fm.classification === "string" ? fm.classification : undefined;
    } catch {
      // Skip tasks we can't read frontmatter for
    }
    out.push({
      id,
      title: entry.title,
      kind: entry.kind,
      status: entry.status as Status,
      tags,
      hasCleanResult: entry.has_clean_result,
      classification,
    });
  }
  out.sort((a, b) => b.id - a.id);
  return out;
}

export function tasksByStatus(): Record<Status, TaskListing[]> {
  const all = listAllTasks();
  const out = {} as Record<Status, TaskListing[]>;
  for (const status of STATUSES) out[status] = [];
  for (const t of all) out[t.status].push(t);
  return out;
}
