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
import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";
import { getTask, resolveTaskPath } from "@/lib/tasks";
import {
  buildBodyEditPrompt,
  classifyIntent,
  readHeadSha,
  runClaudeCodeEdit,
  streamSidecarChat,
  writeTaskBodyUnchecked,
} from "@/lib/claude-comment-ops";

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
  // Optional pointer to a parent comment in the same comments.jsonl. When
  // set, this user comment is a reply within an existing thread (typically
  // a follow-up to a Claude reply). The CommentList renderer uses
  // `in_reply_to` to nest the row under its parent. The anchor (if any)
  // is inherited from the chain root so the entire thread shares one
  // <mark>.
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
 * Thread helpers — read comments.jsonl, walk the in_reply_to chain.
 *
 * Used by POST to (a) validate the parent exists, (b) inherit the chain
 * root's anchor, (c) decide whether to fire a Claude reply (any ancestor
 * authored by Claude or carrying an `@claude` mention), and (d) build
 * the multi-turn context the sidecar prompt needs.
 *
 * Reads are best-effort — malformed lines are skipped (they belong to
 * other tools like save-qa). All callers are inside `withFileLock`.
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
      // Accept either `in_reply_to` (current) or `parent_id` (legacy).
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

/**
 * Walk parent chain from the given comment id up to the chain root.
 * Returns the ordered ancestor list with the root LAST and the
 * immediate-parent FIRST (i.e. ascending toward root). The starting
 * comment itself is NOT included. Tolerates broken links (returns the
 * partial chain) and refuses cycles (max 32 hops).
 */
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

/**
 * The trigger for spawning a Claude reply on a follow-up comment:
 * any ancestor authored by Claude (kind anchor-comment-reply or author
 * "claude"), any ancestor body mentioning @claude, OR the new body
 * itself mentioning @claude. Returns true if Claude should be invoked.
 */
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

/**
 * Build the multi-turn conversation transcript for the sidecar. The
 * sidecar `/chat` accepts a single user-role message, so we serialize
 * the chain into a labeled transcript and append the new user comment
 * at the end. `ancestors` is the parent chain in root-LAST order from
 * ancestorsOf(); we reverse it so the transcript reads top-down.
 */
function buildThreadTranscript(
  ancestors: ThreadRow[],
  newBody: string,
  newAuthor: string,
): string {
  const chronological = [...ancestors].reverse();
  const lines: string[] = [];
  for (const r of chronological) {
    const speaker = r.author === "claude" || r.kind === "anchor-comment-reply" ? "claude" : r.author;
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
  const requestedAnchor = normalizeAnchor(obj.anchor);
  const inReplyToRaw = typeof obj.in_reply_to === "string" ? obj.in_reply_to.trim() : "";
  const inReplyTo = inReplyToRaw || undefined;

  const file = commentsPath(taskId);
  if (!file) return Response.json({ ok: false, error: "task not found" }, { status: 404 });

  // Resolve the thread context. For a reply: the parent must exist;
  // the anchor is inherited from the chain root so the entire thread
  // shares one <mark>. For a top-level comment: use the requested anchor.
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
    // ancestors = [parent, ..., root]
    ancestors = [parentRow, ...ancestorsOf(rows, parentRow.id)];
    // Chain root = last ancestor (= furthest from the new comment).
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

  // Auto-reply gate. Top-level comments fire only on an explicit `@claude`
  // mention. Thread follow-ups also fire if the chain itself is a Claude
  // thread (any ancestor authored by Claude or carrying an `@claude`
  // mention) so the user doesn't have to re-mention on every turn.
  const mentionsClaudeInBody = CLAUDE_MENTION_RE.test(body);
  const isClaudeThread = inReplyTo
    ? shouldFireClaudeForThread(ancestors, body)
    : mentionsClaudeInBody;
  let willReply = false;
  let pendingReplyId: string | undefined;
  if (isClaudeThread) {
    // Strip the `@claude` token from the user's own body before sending
    // to the sidecar — the model doesn't need the mention syntax. The
    // transcript builder leaves the verbatim ancestor bodies alone.
    const promptBody = body.replace(/(^|[^a-z0-9_])@claude\b/gi, "$1").trim();
    // Charge sidecar-chat rate-limit bucket. If exhausted, skip silently
    // — the user comment is already saved.
    const rateLimit = checkRateLimit("sidecar-chat", clientKey(request));
    if (rateLimit.allowed) {
      // Pre-allocate the reply id so the client can show a "Claude is
      // thinking…" placeholder with the same id; the placeholder is
      // naturally replaced when the persisted row lands on poll.
      pendingReplyId = `acr-${randomUUID()}`;
      willReply = true;
      // Capture the editor-cookie state NOW (request-scoped), since
      // `isEditorAuthed()` reads from `next/headers` cookies which don't
      // survive fire-and-forget detachment.
      const editorAuthed = await isEditorAuthed();
      // For a follow-up in a Claude thread, the model needs the full
      // transcript or it will lose context. Pass ancestors verbatim.
      // The user comment is appended by buildThreadTranscript.
      const conversation = inReplyTo
        ? buildThreadTranscript(ancestors, promptBody, user.email)
        : null;
      void spawnClaudeReply({
        file,
        taskId,
        parentId: row.id,
        replyId: pendingReplyId,
        questionBody: promptBody,
        commentId: row.id,
        anchor,
        editorAuthed,
        conversation,
      }).catch((err) => {
        // Never surface to the user — the comment they posted is already saved.
        console.warn("[updates/comment] auto-reply failed:", err);
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
  replyId,
  questionBody,
  commentId,
  anchor,
  editorAuthed,
  conversation,
}: {
  file: string;
  taskId: number;
  parentId: string;
  replyId: string;
  questionBody: string;
  commentId: string;
  anchor?: AnchorPayload;
  editorAuthed: boolean;
  /**
   * When non-null, this is a follow-up comment in an existing Claude
   * thread. The string is the serialized transcript (ancestors + new
   * user message, role-labeled, in chronological order) produced by
   * buildThreadTranscript(). The "answer" path uses it as the user
   * prompt so the model sees prior turns. body-edit / code-edit paths
   * intentionally ignore it: those intents apply to the latest
   * instruction only, and threading them creates ambiguity about which
   * earlier suggestion the new edit should respect.
   */
  conversation: string | null;
}): Promise<void> {
  // Classify the comment via a fast Haiku call. On any failure we fall
  // back to the "answer" path (safest default).
  let intent = await classifyIntent(questionBody);

  // Auth gate for the mutating paths — body-edit and code-edit require
  // the editor cookie. Site-password viewers (Dan) get downgraded to
  // "answer" with a helpful note prepended.
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
      taskId,
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
    await runBodyEditPath({ file, taskId, parentId, replyId, questionBody, anchor });
    return;
  }

  // code-edit
  await runCodeEditPath({ file, parentId, replyId, questionBody, commentId, anchor });
}

async function runAnswerPath({
  file,
  taskId,
  parentId,
  replyId,
  questionBody,
  anchor,
  downgradeNote,
  conversation,
}: {
  file: string;
  taskId: number;
  parentId: string;
  replyId: string;
  questionBody: string;
  anchor?: AnchorPayload;
  downgradeNote: string | null;
  conversation: string | null;
}): Promise<void> {
  const task = getTask(taskId);
  const taskContext = task
    ? `Task #${taskId}: ${task.frontmatter?.title ?? "(no title)"}` +
      `\nStatus: ${task.status}` +
      (task.body ? `\n\nBody excerpt:\n${task.body.slice(0, 4000)}` : "")
    : `Task #${taskId} (body not on disk)`;
  const anchorBlock = anchor
    ? `The mentor highlighted this text in the result body:\n> ${anchor.quote}\n\n`
    : "";
  // Two shapes for the user prompt:
  //   (a) Top-level comment — present the single mentor question.
  //   (b) Thread follow-up — present the full transcript so the model
  //       sees the prior turns it produced. The transcript already ends
  //       with the new user message.
  const prompt = conversation
    ? `You are answering a mentor's follow-up in an anchored-comment thread on task #${taskId} ` +
      `from the EPS dashboard's /updates page. Reply in plain markdown, concise (target <=200 words). ` +
      `No greetings, no signoffs.\n\n` +
      `Context:\n${taskContext}\n\n` +
      anchorBlock +
      `Conversation so far (most recent message last):\n${conversation}\n\n` +
      `Please respond to the most recent user message in the conversation above.`
    : `You are answering a mentor's comment on task #${taskId} from the EPS dashboard's /updates page. ` +
      `Reply in plain markdown, concise (target <=200 words). No greetings, no signoffs.\n\n` +
      `Context:\n${taskContext}\n\n` +
      anchorBlock +
      `Mentor comment:\n${questionBody}`;
  const text = await streamSidecarChat({
    sessionId: `updates-comment-${taskId}`,
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
  taskId,
  parentId,
  replyId,
  questionBody,
  anchor,
}: {
  file: string;
  taskId: number;
  parentId: string;
  replyId: string;
  questionBody: string;
  anchor?: AnchorPayload;
}): Promise<void> {
  const task = getTask(taskId);
  if (!task || typeof task.body !== "string") {
    await persistReply({
      file,
      replyId,
      parentId,
      anchor,
      body: `Couldn't apply body edit: task #${taskId} body not on disk.`,
    });
    return;
  }
  const prompt = buildBodyEditPrompt({
    currentBody: task.body,
    userComment: questionBody,
    taskId,
  });
  const newBody = await streamSidecarChat({
    sessionId: `updates-comment-bodyedit-${taskId}`,
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
  // Strip an opening/closing markdown fence if the model wrapped the
  // whole body in one (sometimes happens despite the prompt).
  const cleaned = stripWholeBodyFence(newBody);
  const write = await writeTaskBodyUnchecked(taskId, cleaned);
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
    await fs.appendFile(file, JSON.stringify(reply) + "\n", "utf8");
  });
}

function stripWholeBodyFence(s: string): string {
  const trimmed = s.trim();
  // Detect ```<lang>\n...\n``` wrapping the whole thing.
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
        // User-authored anchor-comment rows now carry an optional
        // `in_reply_to` pointer (for thread follow-ups). Pass it through
        // so CommentList nests the row under its parent. Accept the
        // legacy `parent_id` alias as well for shape symmetry with the
        // reply row.
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
  // walks the in_reply_to graph to collect the transitive subtree to
  // drop. Pass 2 rewrites the file, keeping every line whose id is not
  // in the drop-set. Author rules:
  //   - User-authored anchor-comment: only the row's author may delete.
  //     Cascade drops the whole subtree (user follow-ups + Claude replies
  //     + their follow-ups, recursively).
  //   - Claude-authored anchor-comment-reply: any signed-in user may
  //     delete (the reply is a side effect of someone's comment, not
  //     durable mentor input). Cascade drops the whole subtree under
  //     the reply too — orphan sub-replies don't render anywhere, so
  //     leaving them would be dead state.
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
      // Pass 1 — parse all rows, find target, check author, build the
      // children-by-parent index needed for transitive cascade.
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
      if (target.kind === "anchor-comment") {
        if (target.author !== user.email) {
          forbidden = true;
          return;
        }
      } else if (target.kind !== "anchor-comment-reply") {
        // Unknown kind (e.g. question/answer/note from other tools) —
        // refuse to touch it.
        forbidden = true;
        return;
      }
      // Walk the subtree (target + all descendants).
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

      // Pass 2 — rewrite the file, dropping any row whose id is in `drop`.
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
