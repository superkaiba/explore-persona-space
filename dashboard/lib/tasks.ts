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
  track?: string;
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
  track: Track;
};

/**
 * Track = which lane a task lives in.
 *   - "experiment": the agent can run it end-to-end (training, eval,
 *     analysis, infra, batch jobs, surveys).
 *   - "human": needs Thomas — think-about / read / decide work.
 */
export type Track = "experiment" | "human";

const HUMAN_KINDS: ReadonlySet<string> = new Set([
  "note",
  "reading",
  "idea",
  "question",
  "decision",
]);

const EXPERIMENT_KINDS: ReadonlySet<string> = new Set([
  "experiment",
  "analysis",
  "infra",
  "batch",
  "survey",
]);

/**
 * Derive a task's track. Precedence:
 *   1. An explicit, valid `track:` frontmatter value wins.
 *   2. Else derive from `kind`: experiment/analysis/infra/batch/survey →
 *      "experiment"; note/reading/idea/question/decision → "human".
 *   3. Else default "experiment".
 */
export function deriveTrack(fm: Frontmatter | undefined, kind: string | undefined): Track {
  const raw = fm?.track;
  if (raw === "experiment" || raw === "human") return raw;
  const k = (kind ?? "").trim().toLowerCase();
  if (HUMAN_KINDS.has(k)) return "human";
  if (EXPERIMENT_KINDS.has(k)) return "experiment";
  return "experiment";
}

export type TaskListing = {
  id: number;
  title: string;
  kind: string;
  status: Status;
  tags: string[];
  hasCleanResult: boolean;
  classification?: string;
  track: Track;
  /** ISO ts the task entered its current status (last `epm:status-changed`
   *  marker; falls back to `created_at` for never-moved tasks). */
  statusChangedAt: string | null;
  /** Follow-up rounds run on this task: distinct `followup_label`s across
   *  `epm:followup-scope` markers + `epm:free-analysis-followup-run` markers. */
  followupCount: number;
  /** Frontmatter `parent_id` when it's a valid number — the board derives
   *  task families (parent/children/siblings) from these edges. */
  parentId: number | null;
  /** ISO ts of the last meaningful update: max(body.md mtime, status-entry
   *  ts). Deliberately NOT events.jsonl mtime — progress markers tick every
   *  few minutes on in-flight tasks and would make "updated" meaningless. */
  lastActivityAt: string | null;
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
  kind:
    | "question"
    | "answer"
    | "followup-proposal"
    | "note"
    | "anchor-comment"
    | "anchor-comment-reply";
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
  const data = fm.data as Frontmatter;
  return {
    id,
    status,
    path: abs,
    frontmatter: data,
    body: isLegacyHtml ? body.replace(LEGACY_SAGAN_CARD_SENTINEL, "").trimStart() : body,
    isLegacyHtml,
    track: deriveTrack(data, data.kind),
  };
}

export type TaskPlan = {
  id: number;
  status: Status;
  filename: string;
  body: string;
};

/**
 * Returns the latest plan body for a task (the file `plans/plan.md` resolves
 * to, e.g. `plans/v2.md`). Returns null if the task has no plans/ directory
 * or the symlink target is missing.
 */
export function getPlan(id: number): TaskPlan | null {
  const abs = resolveTaskPath(id);
  if (!abs) return null;
  const plansDir = path.join(abs, "plans");
  if (!fs.existsSync(plansDir)) return null;
  const symlink = path.join(plansDir, "plan.md");
  let target: string;
  try {
    target = fs.realpathSync(symlink);
  } catch {
    // No plan.md symlink — fall back to the highest v<K>.md present.
    const files = fs
      .readdirSync(plansDir)
      .filter((f) => /^v\d+\.md$/.test(f))
      .sort((a, b) => {
        const na = Number(a.replace(/^v|\.md$/g, ""));
        const nb = Number(b.replace(/^v|\.md$/g, ""));
        return nb - na;
      });
    if (files.length === 0) return null;
    target = path.join(plansDir, files[0]);
  }
  if (!fs.existsSync(target)) return null;
  const body = fs.readFileSync(target, "utf8");
  const status = path.basename(path.dirname(abs)) as Status;
  return {
    id,
    status,
    filename: path.basename(target),
    body,
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
    .map((l) => {
      const c = JSON.parse(l) as TaskComment & { parent_id?: string };
      // Legacy rows used `parent_id`; normalize to `in_reply_to` so the
      // UI's threading logic sees them.
      if (!c.in_reply_to && c.parent_id) c.in_reply_to = c.parent_id;
      return c;
    });
}

/* -------------------------------------------------------------------------- *
 * Event-derived listing stats (status-entry time + follow-up count).
 *
 * events.jsonl across all tasks is ~37MB, and the board force-dynamic page
 * re-renders every 60s per open tab — so stats are cached per file keyed on
 * (mtime, size) and only re-scanned when the file changes. Lines are
 * string-prefiltered; JSON.parse runs only on candidate lines, and the
 * parsed `kind` field is authoritative (embedded marker names inside note
 * strings carry escaped quotes, so the quoted prefilter can't false-match).
 * -------------------------------------------------------------------------- */

type EventStats = {
  lastStatusChangeTs: string | null;
  followupCount: number;
};

const EMPTY_EVENT_STATS: EventStats = { lastStatusChangeTs: null, followupCount: 0 };

const eventStatsCache = new Map<
  string,
  { mtimeMs: number; size: number; stats: EventStats }
>();

function getEventStats(taskAbs: string): EventStats {
  const p = path.join(taskAbs, "events.jsonl");
  let st: fs.Stats;
  try {
    st = fs.statSync(p);
  } catch {
    return EMPTY_EVENT_STATS;
  }
  const cached = eventStatsCache.get(p);
  if (cached && cached.mtimeMs === st.mtimeMs && cached.size === st.size) {
    return cached.stats;
  }
  let raw: string;
  try {
    raw = fs.readFileSync(p, "utf8");
  } catch {
    return EMPTY_EVENT_STATS;
  }
  let lastStatusChangeTs: string | null = null;
  const followupLabels = new Set<string>();
  let unlabeledFollowupScopes = 0;
  let freeAnalysisRuns = 0;
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    const isStatus = line.includes('"epm:status-changed"');
    const isScope = line.includes('"epm:followup-scope"');
    const isFreeAnalysis = line.includes('"epm:free-analysis-followup-run"');
    if (!isStatus && !isScope && !isFreeAnalysis) continue;
    let ev: TaskEvent;
    try {
      ev = JSON.parse(line) as TaskEvent;
    } catch {
      continue; // tolerate a partial trailing line mid-write
    }
    if (ev.kind === "epm:status-changed") {
      if (typeof ev.ts === "string") lastStatusChangeTs = ev.ts;
    } else if (ev.kind === "epm:followup-scope") {
      // Scope-extension reposts reuse the label — count distinct rounds.
      const m =
        typeof ev.note === "string" ? ev.note.match(/followup_label:\s*(\S+)/) : null;
      if (m) followupLabels.add(m[1]);
      else unlabeledFollowupScopes += 1;
    } else if (ev.kind === "epm:free-analysis-followup-run") {
      freeAnalysisRuns += 1;
    }
  }
  const stats: EventStats = {
    lastStatusChangeTs,
    followupCount: followupLabels.size + unlabeledFollowupScopes + freeAnalysisRuns,
  };
  eventStatsCache.set(p, { mtimeMs: st.mtimeMs, size: st.size, stats });
  return stats;
}

/** Normalize a frontmatter `created_at` (string or YAML-parsed Date). */
function toIsoString(v: unknown): string | null {
  if (typeof v === "string" && v.trim()) return v;
  if (v instanceof Date && !Number.isNaN(v.getTime())) return v.toISOString();
  return null;
}

/** Later of two ISO timestamps (null-tolerant, unparseable treated as null). */
function maxIso(a: string | null, b: string | null): string | null {
  const ams = a ? Date.parse(a) : NaN;
  const bms = b ? Date.parse(b) : NaN;
  if (!Number.isFinite(ams)) return Number.isFinite(bms) ? b : null;
  if (!Number.isFinite(bms)) return a;
  return ams >= bms ? a : b;
}

export function listAllTasks(): TaskListing[] {
  const reg = getRegistry();
  const out: TaskListing[] = [];
  for (const [idStr, entry] of Object.entries(reg.tasks)) {
    const id = Number(idStr);
    if (!Number.isFinite(id)) continue;
    if (!STATUSES.includes(entry.status as Status)) continue;
    // Read frontmatter for richer fields
    const abs = path.join(path.dirname(REGISTRY_PATH), "..", entry.path);
    let tags: string[] = [];
    let classification: string | undefined;
    let fm: Frontmatter | undefined;
    let bodyMtime: string | null = null;
    try {
      const bodyPath = path.join(abs, "body.md");
      const raw = fs.readFileSync(bodyPath, "utf8");
      fm = matter(raw).data as Frontmatter;
      tags = Array.isArray(fm.tags) ? fm.tags : [];
      classification = typeof fm.classification === "string" ? fm.classification : undefined;
      bodyMtime = fs.statSync(bodyPath).mtime.toISOString();
    } catch {
      // Skip tasks we can't read frontmatter for
    }
    const eventStats = getEventStats(abs);
    const statusChangedAt = eventStats.lastStatusChangeTs ?? toIsoString(fm?.created_at);
    const rawParent = fm?.parent_id;
    const parentId =
      typeof rawParent === "number" && Number.isFinite(rawParent)
        ? rawParent
        : typeof rawParent === "string" && /^\d+$/.test(rawParent)
          ? Number(rawParent)
          : null;
    out.push({
      id,
      title: entry.title,
      kind: entry.kind,
      status: entry.status as Status,
      tags,
      hasCleanResult: entry.has_clean_result,
      classification,
      track: deriveTrack(fm, entry.kind),
      statusChangedAt,
      followupCount: eventStats.followupCount,
      parentId,
      lastActivityAt: maxIso(bodyMtime, statusChangedAt),
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

/* -------------------------------------------------------------------------- *
 * Updates page support — recent in-flight / freshly-published tasks.
 * -------------------------------------------------------------------------- */

export type UpdateTaskRow = {
  id: number;
  title: string;
  kind: string;
  status: Status;
  hasCleanResult: boolean;
  classification?: string;
  body: string;
  isLegacyHtml: boolean;
  updatedAt: Date;   // mtime of body.md (best proxy for "last change")
  createdAt: Date;   // mtime of the task directory (folder creation)
};

const ACTIVE_STATUSES: ReadonlySet<Status> = new Set([
  "running",
  "verifying",
  "interpreting",
  "reviewing",
  "followups_running",
  "awaiting_promotion",
]);

/**
 * Tasks the /updates page should surface, ordered by most-recent activity.
 *
 * Filter rules:
 *   - any task in {running, verifying, interpreting, reviewing,
 *     followups_running, awaiting_promotion}, or
 *   - any `completed` task with `has_clean_result=true` AND body.md
 *     touched in the last `recentDays` days.
 *
 * Returns up to `limit` rows.
 */
export function recentTasksForUpdates({
  limit = 20,
  recentDays = 14,
}: { limit?: number; recentDays?: number } = {}): UpdateTaskRow[] {
  const cutoff = Date.now() - recentDays * 24 * 60 * 60 * 1000;
  const reg = getRegistry();
  const rows: UpdateTaskRow[] = [];

  for (const [idStr, entry] of Object.entries(reg.tasks)) {
    const id = Number(idStr);
    if (!Number.isFinite(id)) continue;
    const status = entry.status as Status;
    if (!STATUSES.includes(status)) continue;

    const isActive = ACTIVE_STATUSES.has(status);
    const isRecentClean =
      status === "completed" && entry.has_clean_result;
    if (!isActive && !isRecentClean) continue;

    const abs = path.join(path.dirname(REGISTRY_PATH), "..", entry.path);
    const bodyPath = path.join(abs, "body.md");
    let bodyStat: fs.Stats;
    let bodyText: string;
    try {
      bodyStat = fs.statSync(bodyPath);
      bodyText = fs.readFileSync(bodyPath, "utf8");
    } catch {
      continue;
    }

    if (isRecentClean && bodyStat.mtimeMs < cutoff) continue;

    let dirStat: fs.Stats;
    try {
      dirStat = fs.statSync(abs);
    } catch {
      dirStat = bodyStat;
    }

    const fm = matter(bodyText);
    const data = fm.data as Frontmatter;
    const isLegacyHtml = fm.content.includes(LEGACY_SAGAN_CARD_SENTINEL);
    const cleanedBody = isLegacyHtml
      ? fm.content.replace(LEGACY_SAGAN_CARD_SENTINEL, "").trimStart()
      : fm.content;

    rows.push({
      id,
      title: entry.title,
      kind: entry.kind,
      status,
      hasCleanResult: entry.has_clean_result,
      classification:
        typeof data.classification === "string" ? data.classification : undefined,
      body: cleanedBody,
      isLegacyHtml,
      updatedAt: new Date(bodyStat.mtimeMs),
      createdAt: new Date(dirStat.ctimeMs),
    });
  }

  rows.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
  return rows.slice(0, limit);
}
