/**
 * Server-only reader for the session-progress cache.
 *
 * Contract (pinned — another agent writes this file; do not deviate):
 *
 *   ~/.eps-autonomous/session_progress.json
 *   {
 *     "updated_at": "<ISO8601 UTC>",
 *     "sessions": {
 *       "<happy_session_id>": {
 *         "issue": 492,
 *         "status": "planning",
 *         "dir": "explore-persona-space",
 *         "live": true,
 *         "pid": 1637665,
 *         "transcript": "/home/.../<uuid>.jsonl",
 *         "summary": "1-2 sentence freshest \"what it's doing now\"",
 *         "summary_model": "claude-haiku-4-5-20251001",
 *         "summary_ts": "<ISO8601 UTC>",
 *         "last_activity_ts": "<ISO8601 UTC>",
 *         "error": null
 *       }
 *     }
 *   }
 *
 * If the file is missing, empty, or unparseable, return an empty snapshot
 * (NOT a throw) — the writer cron may not have run yet, and the page is
 * supposed to render a clean empty state.
 *
 * Per-session shape is permissive on read: fields are individually
 * narrowed and unknown / malformed fields fall back to safe defaults so a
 * single bad row never blanks the whole page.
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

export const SESSION_PROGRESS_PATH = path.join(
  homedir(),
  ".eps-autonomous",
  "session_progress.json",
);

export type SessionRow = {
  sessionId: string;
  issue: number | null;
  status: string | null;
  dir: string | null;
  live: boolean | null;
  pid: number | null;
  transcript: string | null;
  summary: string | null;
  summaryModel: string | null;
  summaryTs: string | null;
  lastActivityTs: string | null;
  error: string | null;
};

export type SessionSnapshot = {
  /** ISO8601 UTC string from the top-level `updated_at`, or null if the
   * cache file isn't there yet. */
  updatedAt: string | null;
  /** All sessions in the file. Sorting is the caller's job. */
  sessions: SessionRow[];
  /** Reason the snapshot is empty / partial, for diagnostic display.
   * `null` = file present and well-formed (even if `sessions` is empty). */
  readError: string | null;
};

export function loadSessionSnapshot(): SessionSnapshot {
  let raw: string;
  try {
    raw = readFileSync(SESSION_PROGRESS_PATH, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      return { updatedAt: null, sessions: [], readError: null };
    }
    return {
      updatedAt: null,
      sessions: [],
      readError: `failed to read ${SESSION_PROGRESS_PATH}: ${(err as Error).message}`,
    };
  }

  if (raw.trim() === "") {
    return { updatedAt: null, sessions: [], readError: null };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    return {
      updatedAt: null,
      sessions: [],
      readError: `cache JSON unparseable: ${(err as Error).message}`,
    };
  }

  if (!parsed || typeof parsed !== "object") {
    return { updatedAt: null, sessions: [], readError: "cache root is not an object" };
  }

  const obj = parsed as Record<string, unknown>;
  const updatedAt = typeof obj.updated_at === "string" ? obj.updated_at : null;

  const sessionsField = obj.sessions;
  if (!sessionsField || typeof sessionsField !== "object") {
    return { updatedAt, sessions: [], readError: null };
  }

  const sessions: SessionRow[] = [];
  for (const [sessionId, value] of Object.entries(
    sessionsField as Record<string, unknown>,
  )) {
    sessions.push(normalizeSession(sessionId, value));
  }

  return { updatedAt, sessions, readError: null };
}

function normalizeSession(sessionId: string, value: unknown): SessionRow {
  const empty: SessionRow = {
    sessionId,
    issue: null,
    status: null,
    dir: null,
    live: null,
    pid: null,
    transcript: null,
    summary: null,
    summaryModel: null,
    summaryTs: null,
    lastActivityTs: null,
    error: "session entry is not an object",
  };
  if (!value || typeof value !== "object") return empty;
  const o = value as Record<string, unknown>;
  return {
    sessionId,
    issue: typeof o.issue === "number" && Number.isFinite(o.issue) ? o.issue : null,
    status: typeof o.status === "string" ? o.status : null,
    dir: typeof o.dir === "string" ? o.dir : null,
    live: typeof o.live === "boolean" ? o.live : null,
    pid: typeof o.pid === "number" && Number.isFinite(o.pid) ? o.pid : null,
    transcript: typeof o.transcript === "string" ? o.transcript : null,
    summary: typeof o.summary === "string" ? o.summary : null,
    summaryModel: typeof o.summary_model === "string" ? o.summary_model : null,
    summaryTs: typeof o.summary_ts === "string" ? o.summary_ts : null,
    lastActivityTs:
      typeof o.last_activity_ts === "string" ? o.last_activity_ts : null,
    error: typeof o.error === "string" && o.error.length > 0 ? o.error : null,
  };
}
