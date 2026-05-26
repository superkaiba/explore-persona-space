/**
 * Per-card anchored comments for the /updates page.
 *
 *   POST   { taskId, body, anchor?: {quote, prefix, suffix} }  -> { ok, id }
 *   GET    ?taskId=N                                            -> { comments: [...] }
 *   DELETE { taskId, commentId }                                -> { ok }
 *
 * Each row is appended to `tasks/<status>/<id>/comments.jsonl` with
 * `author: <session email>` and `kind: "anchor-comment"`. We append
 * directly via `fs` (no `task.py add-comment` exists) and serialize
 * concurrent writes per-file. DELETE rewrites the file under the same
 * lock and only removes anchor-comment rows the requester authored — so
 * a malicious payload can't nuke saved Q&A or other tools' rows.
 *
 * The shape mirrors the Sagan `/tasks/<id>/CommentableBody` flow so the
 * same `<CommentableBody>` + `<CommentList>` components can render it
 * on the /updates cards.
 *
 * **Auto-reply from Claude.** Each user comment is treated as a question
 * directed at Claude Code. After persisting the user row, we
 * fire-and-forget a streaming sidecar call (using the same path the
 * browser-side MentorClaudePanel uses) and append a second row when the
 * stream finishes:
 *
 *   { id, ts, author: "claude", kind: "anchor-comment-reply",
 *     body: <markdown>, parent_id: <user-row-id>,
 *     anchor: <same anchor as parent if present> }
 *
 * The POST response returns as soon as the user row lands; the reply
 * shows up on the next GET refresh (CardCommentBox polls after submit).
 * Rate-limit budget for the sidecar reply is charged to the same
 * `sidecar-chat` bucket the browser would charge.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { requireSessionAuth } from "@/lib/auth";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";
import { mintSidecarToken } from "@/lib/sidecar-token";
import { getTask, resolveTaskPath } from "@/lib/tasks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_BODY_CHARS = 50_000;
const MAX_ANCHOR_CHARS = 2_000;
const MAX_CONTEXT_CHARS = 200;
// Hard cap on assistant body length we'll persist. Sidecar streams can
// run away (tool loops, runaway codegen). The /tasks/<N>/comments.jsonl
// reader can choke on multi-MB rows — clamp before write.
const MAX_REPLY_CHARS = 40_000;
// How long to wait for the sidecar to emit `done` before giving up and
// persisting whatever we have. The fire-and-forget runs in the route's
// background after the user-response returns, but we still want a hard
// ceiling so a wedged sidecar doesn't leak handles forever.
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
};

type AnchorCommentReplyRow = {
  id: string;
  ts: string;
  author: "claude";
  kind: "anchor-comment-reply";
  body: string;
  in_reply_to: string;
  anchor?: AnchorPayload;
};

/* -------------------------------------------------------------------------- *
 * In-process mutex per comments.jsonl file. Same shape as save-qa/route.ts.
 * -------------------------------------------------------------------------- */

const locks = new Map<string, Promise<void>>();

async function withFileLock<T>(file: string, fn: () => Promise<T>): Promise<T> {
  const prev = locks.get(file) ?? Promise.resolve();
  let release: () => void;
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
    release!();
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

function clampStr(raw: unknown, cap: number): string {
  const s = typeof raw === "string" ? raw : "";
  if (s.length <= cap) return s;
  return s.slice(0, cap);
}

function normalizeAnchor(raw: unknown): AnchorPayload | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const obj = raw as Record<string, unknown>;
  const quote = clampStr(obj.quote, MAX_ANCHOR_CHARS).trim();
  if (!quote) return undefined;
  const out: AnchorPayload = { quote };
  const prefix = clampStr(obj.prefix, MAX_CONTEXT_CHARS);
  const suffix = clampStr(obj.suffix, MAX_CONTEXT_CHARS);
  if (prefix) out.prefix = prefix;
  if (suffix) out.suffix = suffix;
  return out;
}

/* -------------------------------------------------------------------------- *
 * POST — append a new anchor-comment row.
 * -------------------------------------------------------------------------- */

export async function POST(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

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
  const body = String(obj.body ?? "").trim();
  if (!body) return Response.json({ ok: false, error: "body empty" }, { status: 400 });
  if (body.length > MAX_BODY_CHARS) {
    return Response.json(
      { ok: false, error: `body exceeds ${MAX_BODY_CHARS} chars` },
      { status: 413 },
    );
  }
  const anchor = normalizeAnchor(obj.anchor);

  const file = commentsPath(taskId);
  if (!file) return Response.json({ ok: false, error: "task not found" }, { status: 404 });

  const row: AnchorCommentRow = {
    id: `ac-${randomUUID()}`,
    ts: new Date().toISOString(),
    author: user.email,
    kind: "anchor-comment",
    body,
    ...(anchor ? { anchor } : {}),
  };

  try {
    await withFileLock(file, async () => {
      await fs.mkdir(path.dirname(file), { recursive: true });
      await fs.appendFile(file, JSON.stringify(row) + "\n", "utf8");
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return Response.json({ ok: false, error: `append failed: ${msg}` }, { status: 500 });
  }

  // Auto-reply gate: only invoke Claude when the comment body contains
  // an `@claude` mention (case-insensitive, word-boundary). Plain
  // comments stay quiet. This lets Dan leave notes without burning the
  // Anthropic budget on every drive-by remark.
  const mentionsClaude = /(^|[^a-z0-9_])@claude(\b|$)/i.test(body);
  if (mentionsClaude) {
    // Strip the `@claude` token from the prompt before sending to the
    // sidecar — the model doesn't need the mention syntax, just the
    // question.
    const promptBody = body.replace(/(^|[^a-z0-9_])@claude\b/gi, "$1").trim();
    // Charge sidecar-chat rate-limit bucket. If exhausted, skip silently
    // — the user comment is already saved.
    const rateLimit = checkRateLimit("sidecar-chat", clientKey(request));
    if (rateLimit.allowed) {
      void spawnClaudeReply({
        file,
        taskId,
        parentId: row.id,
        questionBody: promptBody,
        anchor,
      }).catch((err) => {
        // Never surface to the user — the comment they posted is already saved.
        // eslint-disable-next-line no-console
        console.warn("[updates/comment] auto-reply failed:", err);
      });
    }
  }

  return Response.json({ ok: true, id: row.id, ts: row.ts });
}

/* -------------------------------------------------------------------------- *
 * Auto-reply helper. Streams the sidecar SSE and persists a single
 * `kind: "anchor-comment-reply"` row when the stream completes (or the
 * REPLY_TIMEOUT_MS fires).
 *
 * Tool-use events are ignored — we only persist the assembled assistant
 * prose. If the sidecar is unconfigured (no `mintSidecarToken` secret /
 * URL) we no-op silently.
 * -------------------------------------------------------------------------- */

async function spawnClaudeReply({
  file,
  taskId,
  parentId,
  questionBody,
  anchor,
}: {
  file: string;
  taskId: number;
  parentId: string;
  questionBody: string;
  anchor?: AnchorPayload;
}): Promise<void> {
  const tokenResult = await mintSidecarToken();
  if (!tokenResult.ok) return;

  const task = getTask(taskId);
  const taskContext = task
    ? `Task #${taskId}: ${task.frontmatter?.title ?? "(no title)"}` +
      `\nStatus: ${task.status}` +
      (task.body ? `\n\nBody excerpt:\n${task.body.slice(0, 4000)}` : "")
    : `Task #${taskId} (body not on disk)`;
  const prompt =
    `You are answering a mentor's comment on task #${taskId} from the EPS dashboard's /updates page. ` +
    `Reply in plain markdown, concise (target <=200 words). No greetings, no signoffs.\n\n` +
    `Context:\n${taskContext}\n\n` +
    (anchor
      ? `The mentor highlighted this text in the result body:\n> ${anchor.quote}\n\n`
      : "") +
    `Mentor comment:\n${questionBody}`;

  const upstream = await fetchWithTimeout(
    `${tokenResult.baseUrl}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${tokenResult.token}`,
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        session_id: `updates-comment-${taskId}`,
        provider: "claude_code",
        messages: [{ role: "user", content: prompt }],
      }),
    },
    REPLY_TIMEOUT_MS,
  );

  if (!upstream.ok || !upstream.body) {
    return;
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let assembled = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const chunks = buf.split(/\r?\n\r?\n/);
      buf = chunks.pop() ?? "";
      for (const eventText of chunks) {
        const parsed = parseSseEventServer(eventText);
        if (!parsed) continue;
        if (parsed.eventName === "token") {
          const t = parsed.data.text;
          if (typeof t === "string") assembled += t;
        } else if (parsed.eventName === "done") {
          // Stream complete — drain any final buffer + break.
        } else if (parsed.eventName === "error") {
          // Swallow — the partial assembled text (if any) still gets posted.
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // No-op — reader may already be detached on timeout.
    }
  }

  const text = assembled.trim();
  if (!text) return;
  const clipped = text.length > MAX_REPLY_CHARS ? text.slice(0, MAX_REPLY_CHARS) : text;

  const reply: AnchorCommentReplyRow = {
    id: `acr-${randomUUID()}`,
    ts: new Date().toISOString(),
    author: "claude",
    kind: "anchor-comment-reply",
    body: clipped,
    in_reply_to: parentId,
    ...(anchor ? { anchor } : {}),
  };

  await withFileLock(file, async () => {
    await fs.appendFile(file, JSON.stringify(reply) + "\n", "utf8");
  });
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function parseSseEventServer(
  eventText: string,
): { eventName: string; data: Record<string, unknown> } | null {
  if (!eventText.trim()) return null;
  let eventName = "message";
  let dataStr = "";
  for (const line of eventText.split(/\r?\n/)) {
    if (line.startsWith("event: ")) eventName = line.slice(7).trim();
    if (line.startsWith("data: ")) dataStr += line.slice(6).trim();
  }
  if (!dataStr) return null;
  try {
    return { eventName, data: JSON.parse(dataStr) as Record<string, unknown> };
  } catch {
    return null;
  }
}

/* -------------------------------------------------------------------------- *
 * GET — list anchor-comment rows for a task (ts-ascending = post order).
 * -------------------------------------------------------------------------- */

export async function GET(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

  const url = new URL(request.url);
  const taskId = validateTaskId(url.searchParams.get("taskId"));
  if (taskId === null) {
    return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });
  }

  const file = commentsPath(taskId);
  if (!file) return Response.json({ ok: true, comments: [] });

  let raw: string;
  try {
    raw = await fs.readFile(file, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return Response.json({ ok: true, comments: [] });
    throw err;
  }

  const comments: Array<AnchorCommentRow | AnchorCommentReplyRow> = [];
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line) as Record<string, unknown>;
      if (typeof parsed.id !== "string" || typeof parsed.body !== "string") continue;
      const anchor = normalizeAnchor(parsed.anchor);
      if (parsed.kind === "anchor-comment") {
        comments.push({
          id: parsed.id,
          ts: typeof parsed.ts === "string" ? parsed.ts : "",
          author: typeof parsed.author === "string" ? parsed.author : "",
          kind: "anchor-comment",
          body: parsed.body,
          ...(anchor ? { anchor } : {}),
        });
      } else if (parsed.kind === "anchor-comment-reply") {
        // Accept either `in_reply_to` (current) or `parent_id` (legacy)
        // so older rows survive normalization.
        const link =
          typeof parsed.in_reply_to === "string"
            ? parsed.in_reply_to
            : typeof parsed.parent_id === "string"
              ? parsed.parent_id
              : "";
        comments.push({
          id: parsed.id,
          ts: typeof parsed.ts === "string" ? parsed.ts : "",
          author: "claude",
          kind: "anchor-comment-reply",
          body: parsed.body,
          in_reply_to: link,
          ...(anchor ? { anchor } : {}),
        });
      }
    } catch {
      // Skip malformed lines — they belong to other tools.
    }
  }

  return Response.json({ ok: true, comments });
}

/* -------------------------------------------------------------------------- *
 * DELETE — remove an anchor-comment row by id. Only the row's author
 * (matched against the session email) may delete; other rows are preserved
 * verbatim so we don't clobber save-qa or analyzer-posted comments.
 * -------------------------------------------------------------------------- */

export async function DELETE(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

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
  const commentId = String(obj.commentId ?? "").trim();
  if (!commentId) {
    return Response.json({ ok: false, error: "commentId empty" }, { status: 400 });
  }

  const file = commentsPath(taskId);
  if (!file) return Response.json({ ok: false, error: "task not found" }, { status: 404 });

  // Two-pass: pass 1 looks the target up to determine permission +
  // whether it's an anchor-comment (which cascade-deletes its replies)
  // or a reply (deletable by any signed-in user — the original author
  // is Claude). Pass 2 rewrites the file.
  let removed = false;
  let forbidden = false;
  try {
    await withFileLock(file, async () => {
      let raw: string;
      try {
        raw = await fs.readFile(file, "utf8");
      } catch (err) {
        const code = (err as NodeJS.ErrnoException).code;
        if (code === "ENOENT") return;
        throw err;
      }
      // Pass 1 — find target + decide cascade.
      let cascadeFor: string | null = null;
      for (const line of raw.split("\n")) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line) as {
            id?: unknown;
            kind?: unknown;
            author?: unknown;
          };
          if (parsed.id !== commentId) continue;
          if (parsed.kind === "anchor-comment") {
            if (typeof parsed.author === "string" && parsed.author === user.email) {
              cascadeFor = commentId;
            } else {
              forbidden = true;
            }
          } else if (parsed.kind === "anchor-comment-reply") {
            // Replies are authored by Claude — any signed-in user may
            // delete them (they're a side effect of the user's own
            // comment, not durable mentor input).
            cascadeFor = "__reply_only__";
          }
        } catch {
          // ignore
        }
      }
      if (forbidden && cascadeFor === null) return;

      const kept: string[] = [];
      for (const line of raw.split("\n")) {
        if (!line.trim()) continue;
        let drop = false;
        try {
          const parsed = JSON.parse(line) as {
            id?: unknown;
            kind?: unknown;
            in_reply_to?: unknown;
            parent_id?: unknown;
          };
          // Accept either `in_reply_to` (current) or `parent_id` (legacy)
          // so cascade deletion catches both shapes.
          const link =
            typeof parsed.in_reply_to === "string"
              ? parsed.in_reply_to
              : typeof parsed.parent_id === "string"
                ? parsed.parent_id
                : null;
          if (cascadeFor === "__reply_only__") {
            if (
              parsed.id === commentId &&
              parsed.kind === "anchor-comment-reply"
            ) {
              drop = true;
            }
          } else if (cascadeFor) {
            if (parsed.id === cascadeFor) {
              drop = true;
            } else if (
              parsed.kind === "anchor-comment-reply" &&
              link === cascadeFor
            ) {
              drop = true;
            }
          }
        } catch {
          // Preserve unparseable lines verbatim.
        }
        if (drop) {
          removed = true;
          continue;
        }
        kept.push(line);
      }
      if (!removed) return;
      const next = kept.length === 0 ? "" : kept.join("\n") + "\n";
      await fs.writeFile(file, next, "utf8");
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return Response.json({ ok: false, error: `delete failed: ${msg}` }, { status: 500 });
  }

  if (forbidden && !removed) {
    return Response.json(
      { ok: false, error: "not your comment" },
      { status: 403 },
    );
  }
  return Response.json({ ok: true, removed });
}
