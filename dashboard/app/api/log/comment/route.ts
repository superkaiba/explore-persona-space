/**
 * Per-card anchored comments for the /log page.
 *
 *   POST   { entryId, body, anchor?, in_reply_to? }  -> { ok, id }
 *   GET    ?entryId=daily-2026-05-26                  -> { comments: [...] }
 *   DELETE { entryId, commentId }                     -> { ok }
 *
 * Mirror of `app/api/updates/comment/route.ts`. The only differences are
 * that the entity key is a string (`entryId`) validated against the
 * `/^(daily|weekly|ideation)-[\w-]+$/` regex, the comments file lives at
 * `<repo>/logs/comments/<entryId>.jsonl`, and the body-edit path edits a
 * log entry markdown file via `writeLogEntryBody` (preserving
 * frontmatter) instead of `writeTaskBodyUnchecked`.
 *
 * Everything else — anchors, threading via `in_reply_to`, @claude
 * auto-reply gate, intent classifier (answer / body-edit / code-edit),
 * rate-limiting, body/reply size caps, DELETE with author check +
 * transitive subtree cascade — is the same as updates/comment. The two
 * routes share `claude-comment-ops.ts` for the heavy lifting.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";
import { REPO_ROOT } from "@/lib/repo";
import {
  getLogEntry,
  isValidEntryId,
  writeLogEntryBody,
} from "@/lib/logs";
import {
  buildBodyEditPrompt,
  classifyIntent,
  readHeadSha,
  runClaudeCodeEdit,
  streamSidecarChat,
} from "@/lib/claude-comment-ops";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_BODY_CHARS = 50_000;
const MAX_ANCHOR_CHARS = 2_000;
const MAX_CONTEXT_CHARS = 200;
// Hard cap on assistant body length we'll persist. Sidecar streams can
// run away (tool loops, runaway codegen). The CommentList reader can
// choke on multi-MB rows — clamp before write.
const MAX_REPLY_CHARS = 40_000;
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
  in_reply_to?: string;
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
 * In-process mutex per comments.jsonl file. Same shape as updates/comment.
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

function validateEntryId(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  return isValidEntryId(raw) ? raw : null;
}

function commentsPath(entryId: string): string {
  // <repo>/logs/comments/<entryId>.jsonl. entryId is regex-validated by
  // validateEntryId so it can't contain path separators or `..`.
  return path.join(REPO_ROOT, "logs", "comments", `${entryId}.jsonl`);
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
 * Thread helpers — read comments.jsonl, walk the in_reply_to chain.
 *
 * (Identical to updates/comment; copied rather than extracted because the
 * two routes have nearly-but-not-quite-identical row shapes and shipping
 * a shared module now would lock the shape prematurely.)
 * -------------------------------------------------------------------------- */

type ThreadRow = {
  id: string;
  ts: string;
  author: string;
  kind: string;
  body: string;
  in_reply_to?: string;
  anchor?: AnchorPayload;
};

async function readThreadRows(file: string): Promise<ThreadRow[]> {
  let raw: string;
  try {
    raw = await fs.readFile(file, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return [];
    throw err;
  }
  const rows: ThreadRow[] = [];
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line) as Record<string, unknown>;
      if (typeof parsed.id !== "string") continue;
      if (typeof parsed.body !== "string") continue;
      const link =
        typeof parsed.in_reply_to === "string"
          ? parsed.in_reply_to
          : typeof parsed.parent_id === "string"
            ? (parsed.parent_id as string)
            : undefined;
      rows.push({
        id: parsed.id,
        ts: typeof parsed.ts === "string" ? parsed.ts : "",
        author: typeof parsed.author === "string" ? parsed.author : "",
        kind: typeof parsed.kind === "string" ? parsed.kind : "",
        body: parsed.body,
        ...(link ? { in_reply_to: link } : {}),
        ...(normalizeAnchor(parsed.anchor)
          ? { anchor: normalizeAnchor(parsed.anchor) }
          : {}),
      });
    } catch {
      // Skip malformed lines.
    }
  }
  return rows;
}

function ancestorsOf(rows: ThreadRow[], startId: string): ThreadRow[] {
  const byId = new Map(rows.map((r) => [r.id, r]));
  const out: ThreadRow[] = [];
  let cur = byId.get(startId);
  const seen = new Set<string>();
  let hops = 0;
  while (cur && cur.in_reply_to && hops < 32) {
    if (seen.has(cur.id)) break;
    seen.add(cur.id);
    const parent = byId.get(cur.in_reply_to);
    if (!parent) break;
    out.push(parent);
    cur = parent;
    hops += 1;
  }
  return out;
}

const CLAUDE_MENTION_RE = /(^|[^a-z0-9_])@claude(\b|$)/i;

function shouldFireClaudeForThread(
  ancestors: ThreadRow[],
  newBody: string,
): boolean {
  if (CLAUDE_MENTION_RE.test(newBody)) return true;
  for (const a of ancestors) {
    if (a.author === "claude" || a.kind === "anchor-comment-reply") return true;
    if (CLAUDE_MENTION_RE.test(a.body)) return true;
  }
  return false;
}

function buildThreadTranscript(
  ancestors: ThreadRow[],
  newBody: string,
  newAuthor: string,
): string {
  const chronological = [...ancestors].reverse();
  const lines: string[] = [];
  for (const r of chronological) {
    const speaker =
      r.author === "claude" || r.kind === "anchor-comment-reply" ? "claude" : r.author;
    const ts = r.ts || "?";
    lines.push(`[${speaker} @ ${ts}]: ${r.body}`);
  }
  lines.push(`[${newAuthor} @ now]: ${newBody}`);
  return lines.join("\n\n");
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
  const entryId = validateEntryId(obj.entryId);
  if (!entryId) {
    return Response.json({ ok: false, error: "invalid entryId" }, { status: 400 });
  }
  const body = String(obj.body ?? "").trim();
  if (!body) return Response.json({ ok: false, error: "body empty" }, { status: 400 });
  if (body.length > MAX_BODY_CHARS) {
    return Response.json(
      { ok: false, error: `body exceeds ${MAX_BODY_CHARS} chars` },
      { status: 413 },
    );
  }
  const requestedAnchor = normalizeAnchor(obj.anchor);
  const inReplyToRaw = typeof obj.in_reply_to === "string" ? obj.in_reply_to.trim() : "";
  const inReplyTo = inReplyToRaw || undefined;

  // Refuse to host comments on a log entry that doesn't exist on disk.
  // The dashboard's `lib/logs.ts` is the source of truth for what
  // entries are visible / addressable.
  const entry = await getLogEntry(entryId);
  if (!entry) {
    return Response.json({ ok: false, error: "log entry not found" }, { status: 404 });
  }

  const file = commentsPath(entryId);

  let anchor = requestedAnchor;
  let ancestors: ThreadRow[] = [];
  let parentRow: ThreadRow | undefined;
  if (inReplyTo) {
    const rows = await readThreadRows(file);
    parentRow = rows.find((r) => r.id === inReplyTo);
    if (!parentRow) {
      return Response.json(
        { ok: false, error: "in_reply_to: parent not found" },
        { status: 404 },
      );
    }
    ancestors = [parentRow, ...ancestorsOf(rows, parentRow.id)];
    const rootRow = ancestors[ancestors.length - 1];
    anchor = rootRow?.anchor ?? parentRow.anchor ?? requestedAnchor;
  }

  const row: AnchorCommentRow = {
    id: `ac-${randomUUID()}`,
    ts: new Date().toISOString(),
    author: user.email,
    kind: "anchor-comment",
    body,
    ...(anchor ? { anchor } : {}),
    ...(inReplyTo ? { in_reply_to: inReplyTo } : {}),
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

  const mentionsClaudeInBody = CLAUDE_MENTION_RE.test(body);
  const isClaudeThread = inReplyTo
    ? shouldFireClaudeForThread(ancestors, body)
    : mentionsClaudeInBody;
  let willReply = false;
  let pendingReplyId: string | undefined;
  if (isClaudeThread) {
    const promptBody = body.replace(/(^|[^a-z0-9_])@claude\b/gi, "$1").trim();
    const rateLimit = checkRateLimit("sidecar-chat", clientKey(request));
    if (rateLimit.allowed) {
      pendingReplyId = `acr-${randomUUID()}`;
      willReply = true;
      const editorAuthed = await isEditorAuthed();
      const conversation = inReplyTo
        ? buildThreadTranscript(ancestors, promptBody, user.email)
        : null;
      void spawnClaudeReply({
        file,
        entryId,
        parentId: row.id,
        replyId: pendingReplyId,
        questionBody: promptBody,
        commentId: row.id,
        anchor,
        editorAuthed,
        conversation,
      }).catch((err) => {
        console.warn("[log/comment] auto-reply failed:", err);
      });
    }
  }

  return Response.json({
    ok: true,
    id: row.id,
    ts: row.ts,
    ...(willReply && pendingReplyId
      ? { will_reply: true, pending_reply_id: pendingReplyId }
      : {}),
  });
}

/* -------------------------------------------------------------------------- *
 * Auto-reply helper — same three-intent dispatch as updates/comment.
 *
 * The body-edit path swaps the task-flavored helpers for log-flavored
 * ones: `getLogEntry(entryId)` for the current body and
 * `writeLogEntryBody(entryId, newBody)` for the apply step. Frontmatter
 * is preserved by `writeLogEntryBody`.
 * -------------------------------------------------------------------------- */

async function spawnClaudeReply({
  file,
  entryId,
  parentId,
  replyId,
  questionBody,
  commentId,
  anchor,
  editorAuthed,
  conversation,
}: {
  file: string;
  entryId: string;
  parentId: string;
  replyId: string;
  questionBody: string;
  commentId: string;
  anchor?: AnchorPayload;
  editorAuthed: boolean;
  conversation: string | null;
}): Promise<void> {
  let intent = await classifyIntent(questionBody);

  let downgradeNote: string | null = null;
  if ((intent === "body-edit" || intent === "code-edit") && !editorAuthed) {
    downgradeNote =
      "I'd suggest applying this change, but I don't have edit permission on the dashboard. " +
      "Ask Thomas (or use the Edit button on the card) to apply this.";
    intent = "answer";
  }

  if (intent === "answer") {
    await runAnswerPath({
      file,
      entryId,
      parentId,
      replyId,
      questionBody,
      anchor,
      downgradeNote,
      conversation,
    });
    return;
  }

  if (intent === "body-edit") {
    await runBodyEditPath({ file, entryId, parentId, replyId, questionBody, anchor });
    return;
  }

  // code-edit — same dashboard-edit path as updates/comment.
  await runCodeEditPath({ file, parentId, replyId, questionBody, commentId, anchor });
}

async function runAnswerPath({
  file,
  entryId,
  parentId,
  replyId,
  questionBody,
  anchor,
  downgradeNote,
  conversation,
}: {
  file: string;
  entryId: string;
  parentId: string;
  replyId: string;
  questionBody: string;
  anchor?: AnchorPayload;
  downgradeNote: string | null;
  conversation: string | null;
}): Promise<void> {
  const entry = await getLogEntry(entryId);
  const entryContext = entry
    ? `Log entry ${entryId} (kind: ${entry.kind}, date: ${entry.date}): ${entry.title}` +
      (entry.body ? `\n\nBody excerpt:\n${entry.body.slice(0, 4000)}` : "")
    : `Log entry ${entryId} (body not on disk)`;
  const anchorBlock = anchor
    ? `The mentor highlighted this text in the entry body:\n> ${anchor.quote}\n\n`
    : "";
  const prompt = conversation
    ? `You are answering a mentor's follow-up in an anchored-comment thread on log entry ${entryId} ` +
      `from the EPS dashboard's /log page. Reply in plain markdown, concise (target <=200 words). ` +
      `No greetings, no signoffs.\n\n` +
      `Context:\n${entryContext}\n\n` +
      anchorBlock +
      `Conversation so far (most recent message last):\n${conversation}\n\n` +
      `Please respond to the most recent user message in the conversation above.`
    : `You are answering a mentor's comment on log entry ${entryId} from the EPS dashboard's /log page. ` +
      `Reply in plain markdown, concise (target <=200 words). No greetings, no signoffs.\n\n` +
      `Context:\n${entryContext}\n\n` +
      anchorBlock +
      `Mentor comment:\n${questionBody}`;
  const text = await streamSidecarChat({
    sessionId: `log-comment-${entryId}`,
    prompt,
    timeoutMs: REPLY_TIMEOUT_MS,
    maxChars: MAX_REPLY_CHARS,
  });
  if (!text) return;
  const finalBody = downgradeNote ? `${downgradeNote}\n\n${text}` : text;
  await persistReply({ file, replyId, parentId, anchor, body: finalBody });
}

async function runBodyEditPath({
  file,
  entryId,
  parentId,
  replyId,
  questionBody,
  anchor,
}: {
  file: string;
  entryId: string;
  parentId: string;
  replyId: string;
  questionBody: string;
  anchor?: AnchorPayload;
}): Promise<void> {
  const entry = await getLogEntry(entryId);
  if (!entry) {
    await persistReply({
      file,
      replyId,
      parentId,
      anchor,
      body: `Couldn't apply body edit: log entry ${entryId} not on disk.`,
    });
    return;
  }
  // buildBodyEditPrompt is task-numbered in its template, but the prose
  // it produces is generic enough that pointing it at a log entry still
  // works. The fake numeric `taskId` slot is purely an internal label
  // for the prompt — the model sees "task #<entryId>" which is harmless.
  const prompt = buildBodyEditPrompt({
    currentBody: entry.body,
    userComment: questionBody,
    // Pass 0 to signal "not a task". The prompt text references it but
    // the model is told to return a verbatim body, so the actual id is
    // immaterial — what matters is the currentBody + userComment.
    taskId: 0,
  })
    // Replace the misleading "task #0" mention with the actual entryId
    // so the model's mental model of what it's editing is correct.
    .replace(/task #0\b/g, `log entry ${entryId}`);
  const newBody = await streamSidecarChat({
    sessionId: `log-comment-bodyedit-${entryId}`,
    prompt,
    timeoutMs: REPLY_TIMEOUT_MS,
    maxChars: 1_000_000,
  });
  if (!newBody) {
    await persistReply({
      file,
      replyId,
      parentId,
      anchor,
      body: "Couldn't apply body edit: sidecar returned no content.",
    });
    return;
  }
  const cleaned = stripWholeBodyFence(newBody);
  const write = await writeLogEntryBody(entryId, cleaned);
  if (!write.ok) {
    await persistReply({
      file,
      replyId,
      parentId,
      anchor,
      body: `Couldn't apply body edit: ${write.error}`,
    });
    return;
  }
  const sha = await readHeadSha();
  const summary = oneLineSummary(questionBody);
  await persistReply({
    file,
    replyId,
    parentId,
    anchor,
    body: `Edited body. ${summary} Commit \`${sha}\`.`,
  });
}

async function runCodeEditPath({
  file,
  parentId,
  replyId,
  questionBody,
  commentId,
  anchor,
}: {
  file: string;
  parentId: string;
  replyId: string;
  questionBody: string;
  commentId: string;
  anchor?: AnchorPayload;
}): Promise<void> {
  const result = await runClaudeCodeEdit({
    userComment: questionBody,
    commentId,
  });
  let body: string;
  if (result.ok) {
    body = [
      `Applied dashboard edit. Build OK. Restarted service.`,
      `Commit \`${result.sha}\`. ${result.summary}`,
    ].join(" ");
  } else {
    body = [
      `Couldn't apply code edit: ${result.error}`,
      result.tail ? `\n\nBuild/diff tail:\n\`\`\`\n${result.tail}\n\`\`\`` : "",
    ].join("");
  }
  await persistReply({ file, replyId, parentId, anchor, body });
}

async function persistReply({
  file,
  replyId,
  parentId,
  anchor,
  body,
}: {
  file: string;
  replyId: string;
  parentId: string;
  anchor?: AnchorPayload;
  body: string;
}): Promise<void> {
  const clipped = body.length > MAX_REPLY_CHARS ? body.slice(0, MAX_REPLY_CHARS) : body;
  const reply: AnchorCommentReplyRow = {
    id: replyId,
    ts: new Date().toISOString(),
    author: "claude",
    kind: "anchor-comment-reply",
    body: clipped,
    in_reply_to: parentId,
    ...(anchor ? { anchor } : {}),
  };
  await withFileLock(file, async () => {
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.appendFile(file, JSON.stringify(reply) + "\n", "utf8");
  });
}

function stripWholeBodyFence(s: string): string {
  const trimmed = s.trim();
  const m = trimmed.match(/^```(?:[a-zA-Z0-9_-]*)?\n([\s\S]*)\n```$/);
  if (m) return m[1].trim();
  return trimmed;
}

function oneLineSummary(userComment: string): string {
  const compact = userComment.replace(/\s+/g, " ").trim();
  const cap = 200;
  const slice = compact.length > cap ? compact.slice(0, cap) + "…" : compact;
  return `Requested: ${slice}`;
}

/* -------------------------------------------------------------------------- *
 * GET — list anchor-comment rows for a log entry (ts-ascending = post order).
 * -------------------------------------------------------------------------- */

export async function GET(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

  const url = new URL(request.url);
  const entryId = validateEntryId(url.searchParams.get("entryId"));
  if (!entryId) {
    return Response.json({ ok: false, error: "invalid entryId" }, { status: 400 });
  }

  const file = commentsPath(entryId);
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
        const link =
          typeof parsed.in_reply_to === "string"
            ? parsed.in_reply_to
            : typeof parsed.parent_id === "string"
              ? parsed.parent_id
              : "";
        comments.push({
          id: parsed.id,
          ts: typeof parsed.ts === "string" ? parsed.ts : "",
          author: typeof parsed.author === "string" ? parsed.author : "",
          kind: "anchor-comment",
          body: parsed.body,
          ...(link ? { in_reply_to: link } : {}),
          ...(anchor ? { anchor } : {}),
        });
      } else if (parsed.kind === "anchor-comment-reply") {
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
 * DELETE — remove an anchor-comment row by id. Same author-gating +
 * transitive subtree cascade as updates/comment.
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
  const entryId = validateEntryId(obj.entryId);
  if (!entryId) {
    return Response.json({ ok: false, error: "invalid entryId" }, { status: 400 });
  }
  const commentId = String(obj.commentId ?? "").trim();
  if (!commentId) {
    return Response.json({ ok: false, error: "commentId empty" }, { status: 400 });
  }

  const file = commentsPath(entryId);
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
      type ParsedRow = {
        id: string;
        kind?: string;
        author?: string;
        parent?: string;
      };
      const rows: ParsedRow[] = [];
      const childrenOf = new Map<string, string[]>();
      let target: ParsedRow | null = null;
      for (const line of raw.split("\n")) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line) as Record<string, unknown>;
          if (typeof parsed.id !== "string") continue;
          const link =
            typeof parsed.in_reply_to === "string"
              ? parsed.in_reply_to
              : typeof parsed.parent_id === "string"
                ? (parsed.parent_id as string)
                : undefined;
          const row: ParsedRow = {
            id: parsed.id,
            kind: typeof parsed.kind === "string" ? parsed.kind : undefined,
            author: typeof parsed.author === "string" ? parsed.author : undefined,
            parent: link,
          };
          rows.push(row);
          if (row.parent) {
            const arr = childrenOf.get(row.parent) ?? [];
            arr.push(row.id);
            childrenOf.set(row.parent, arr);
          }
          if (row.id === commentId) target = row;
        } catch {
          // Skip malformed lines for the index — Pass 2 preserves them.
        }
      }
      if (!target) return;
      // TS narrows `target` to never after the `=== null` guard above; the
      // type-narrowing through the closure loop is the issue. Cast back.
      const found = target as ParsedRow;
      if (found.kind === "anchor-comment") {
        if (found.author !== user.email) {
          forbidden = true;
          return;
        }
      } else if (found.kind !== "anchor-comment-reply") {
        forbidden = true;
        return;
      }
      const drop = new Set<string>();
      const stack = [commentId];
      let hops = 0;
      while (stack.length && hops < 1024) {
        const cid = stack.pop()!;
        if (drop.has(cid)) continue;
        drop.add(cid);
        const kids = childrenOf.get(cid) ?? [];
        for (const k of kids) stack.push(k);
        hops += 1;
      }

      const kept: string[] = [];
      for (const line of raw.split("\n")) {
        if (!line.trim()) continue;
        let dropLine = false;
        try {
          const parsed = JSON.parse(line) as { id?: unknown };
          if (typeof parsed.id === "string" && drop.has(parsed.id)) {
            dropLine = true;
          }
        } catch {
          // Preserve unparseable lines verbatim.
        }
        if (dropLine) {
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
