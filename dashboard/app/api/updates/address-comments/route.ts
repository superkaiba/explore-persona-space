/**
 * POST /api/updates/address-comments — Change D: "Address comments"
 * button on the /updates modal full-view.
 *
 * Loads the current task body + every unaddressed `anchor-comment` row,
 * asks Claude (via the local sidecar) to rewrite the body to address
 * the comments, then:
 *   1. saves the new body (commits via `task.py set-body`),
 *   2. rewrites `comments.jsonl` to mark each addressed row with
 *      `addressed: true`, `addressed_in: <new sha>`, `addressed_note: <claude note>`,
 *   3. appends a synthesis `kind: anchor-comment-reply` row linked to
 *      the first addressed comment.
 *
 * Auth: `isEditorAuthed()`-gated. Site-password viewers get 401.
 * Concurrency: serialized per-file via the same in-process mutex
 * `/api/updates/comment` uses (replicated here to keep the modules
 * decoupled).
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { getTask, resolveTaskPath } from "@/lib/tasks";
import {
  parseAddressJson,
  readHeadSha,
  streamSidecarChat,
  writeTaskBodyUnchecked,
} from "@/lib/claude-comment-ops";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_BODY_CHARS = 1_000_000;
const REPLY_TIMEOUT_MS = 5 * 60 * 1000;

type AnchorPayload = {
  quote: string;
  prefix?: string;
  suffix?: string;
};

type AnchorCommentRow = {
  id: string;
  ts: string;
  author: string;
  kind: "anchor-comment";
  body: string;
  anchor?: AnchorPayload;
  addressed?: boolean;
  addressed_in?: string;
  addressed_note?: string;
};

const locks = new Map<string, Promise<void>>();

async function withFileLock<T>(file: string, fn: () => Promise<T>): Promise<T> {
  const prev = locks.get(file) ?? Promise.resolve();
  let release: () => void = () => {};
  const next = new Promise<void>((resolve) => {
    release = resolve;
  });
  locks.set(
    file,
    prev.then(() => next),
  );
  await prev;
  try {
    return await fn();
  } finally {
    release();
    if (locks.get(file) === next) locks.delete(file);
  }
}

function validateTaskId(raw: unknown): number | null {
  const n = Number(raw);
  return Number.isFinite(n) && Number.isInteger(n) && n >= 1 ? n : null;
}

function commentsPath(taskId: number): string | null {
  const dir = resolveTaskPath(taskId);
  if (!dir) return null;
  return path.join(dir, "comments.jsonl");
}

export async function POST(request: Request) {
  const user = await requireSessionAuth();
  if (!user) {
    return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });
  }
  if (!(await isEditorAuthed())) {
    return Response.json(
      { ok: false, error: "editor cookie required" },
      { status: 403 },
    );
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ ok: false, error: "invalid json" }, { status: 400 });
  }
  const obj = (payload ?? {}) as Record<string, unknown>;
  const taskId = validateTaskId(obj.taskId);
  if (taskId === null) {
    return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });
  }

  const file = commentsPath(taskId);
  if (!file) {
    return Response.json({ ok: false, error: "task not found" }, { status: 404 });
  }

  const task = getTask(taskId);
  if (!task || typeof task.body !== "string") {
    return Response.json(
      { ok: false, error: "task body not on disk" },
      { status: 404 },
    );
  }
  const currentBody = task.body;
  if (currentBody.length > MAX_BODY_CHARS) {
    return Response.json(
      { ok: false, error: "body too large" },
      { status: 413 },
    );
  }

  // Load comments.jsonl, pick unaddressed anchor-comment rows.
  const openComments = await readOpenComments(file);
  if (openComments.length === 0) {
    return Response.json(
      { ok: false, error: "no open comments to address" },
      { status: 409 },
    );
  }

  // Build the prompt per the spec — body + comment list + instruction
  // to return revised body + JSON tail.
  const prompt = [
    `You are revising the body of EPS experiment task #${taskId} to address mentor review comments.`,
    "Return the FULL revised markdown body, then on a new line a single JSON object",
    'of the form: {"addressed": {"<comment_id>": "<one-sentence note on how you addressed it>", ...}}.',
    "Include every comment_id you actually addressed. No commentary outside those two parts.",
    "No code fences around the whole body.",
    "",
    "Current body:",
    currentBody,
    "",
    "Open comments:",
    ...openComments.map((c) => {
      const quote = c.anchor?.quote ? `\n  quote: ${truncate(c.anchor.quote, 400)}` : "";
      return `- id: ${c.id}\n  author: ${c.author}${quote}\n  body: ${truncate(c.body, 1200)}`;
    }),
  ].join("\n");

  const raw = await streamSidecarChat({
    sessionId: `updates-address-${taskId}`,
    prompt,
    timeoutMs: REPLY_TIMEOUT_MS,
    maxChars: 1_200_000,
  });
  if (!raw) {
    return Response.json(
      { ok: false, error: "sidecar returned no content" },
      { status: 502 },
    );
  }
  const parsed = parseAddressJson(raw);
  if (!parsed) {
    return Response.json(
      { ok: false, error: "could not parse JSON tail from sidecar response" },
      { status: 502 },
    );
  }
  const { body: newBody, addressed: addressedMap } = parsed;

  const write = await writeTaskBodyUnchecked(taskId, newBody);
  if (!write.ok) {
    return Response.json(
      { ok: false, error: `task.py set-body failed: ${write.error}` },
      { status: 500 },
    );
  }
  const sha = await readHeadSha();

  // Apply addressed marker to matched rows. IDs in `addressedMap`
  // not present in `openComments` are ignored (model hallucinated id).
  const knownIds = new Set(openComments.map((c) => c.id));
  const appliedIds: string[] = [];
  const appliedNotes: Array<{ id: string; note: string }> = [];
  for (const [id, note] of Object.entries(addressedMap)) {
    if (knownIds.has(id)) {
      appliedIds.push(id);
      appliedNotes.push({ id, note });
    }
  }

  await withFileLock(file, async () => {
    let rawJsonl: string;
    try {
      rawJsonl = await fs.readFile(file, "utf8");
    } catch {
      rawJsonl = "";
    }
    const noteById = new Map(appliedNotes.map((x) => [x.id, x.note]));
    const lines = rawJsonl.split("\n");
    const out: string[] = [];
    for (const line of lines) {
      if (!line.trim()) {
        out.push(line);
        continue;
      }
      let row: Record<string, unknown>;
      try {
        row = JSON.parse(line) as Record<string, unknown>;
      } catch {
        out.push(line);
        continue;
      }
      const id = typeof row.id === "string" ? row.id : null;
      if (id && noteById.has(id) && row.kind === "anchor-comment") {
        row.addressed = true;
        row.addressed_in = sha;
        row.addressed_note = noteById.get(id);
        out.push(JSON.stringify(row));
      } else {
        out.push(line);
      }
    }
    // Append synthesis reply linked to the first addressed comment.
    if (appliedIds.length > 0) {
      const synthesis = {
        id: `acr-${randomUUID()}`,
        ts: new Date().toISOString(),
        author: "claude",
        kind: "anchor-comment-reply",
        body: buildSynthesis(appliedNotes, sha),
        in_reply_to: appliedIds[0],
      };
      if (out.length > 0 && out[out.length - 1] !== "") out.push(JSON.stringify(synthesis));
      else out[out.length - 1] = JSON.stringify(synthesis);
      out.push("");
    }
    const next = out.filter((l, i) => l !== "" || i < out.length - 1).join("\n");
    await fs.writeFile(file, next.endsWith("\n") ? next : next + "\n", "utf8");
  });

  return Response.json({
    ok: true,
    addressed: appliedIds,
    sha,
    body: newBody,
  });
}

async function readOpenComments(file: string): Promise<AnchorCommentRow[]> {
  let raw: string;
  try {
    raw = await fs.readFile(file, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return [];
    throw err;
  }
  const out: AnchorCommentRow[] = [];
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (parsed.kind !== "anchor-comment") continue;
    if (parsed.addressed === true) continue;
    const id = typeof parsed.id === "string" ? parsed.id : "";
    const body = typeof parsed.body === "string" ? parsed.body : "";
    if (!id || !body) continue;
    const anchorRaw = parsed.anchor;
    let anchor: AnchorPayload | undefined;
    if (anchorRaw && typeof anchorRaw === "object") {
      const q = (anchorRaw as { quote?: unknown }).quote;
      if (typeof q === "string" && q.trim()) {
        anchor = { quote: q };
      }
    }
    out.push({
      id,
      ts: typeof parsed.ts === "string" ? parsed.ts : "",
      author: typeof parsed.author === "string" ? parsed.author : "",
      kind: "anchor-comment",
      body,
      ...(anchor ? { anchor } : {}),
    });
  }
  return out;
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + "…" : s;
}

function buildSynthesis(
  notes: Array<{ id: string; note: string }>,
  sha: string,
): string {
  const list = notes
    .slice(0, 12)
    .map((n) => `- \`${n.id}\`: ${n.note}`)
    .join("\n");
  return [
    `Addressed ${notes.length} comment${notes.length === 1 ? "" : "s"} in commit \`${sha}\`:`,
    "",
    list,
  ].join("\n");
}
