/**
 * Per-doc anchored comments for /docs/<slug>.
 *
 *   POST   { slug, body, quote?, section_id?, section_label?, in_reply_to? }
 *                                                            -> { ok, id }
 *   GET    ?slug=<slug>                                      -> { ok, comments: [...] }
 *   DELETE { slug, commentId }                              -> { ok }
 *
 * Rows append to the doc's gitignored `.comments/<stem>.jsonl` (see
 * lib/doc-comments.ts — routed through the shared docs resolver so virtual
 * mentor-update / activity / idea slugs land next to the right file).
 *
 * `quote` carries the highlight-to-comment anchor: the exact selected text
 * MarkdownDoc wraps in <mark> on the rendered body. POST/DELETE require a
 * signed session; DELETE only removes rows the requester authored. GET is
 * open (the docs pages are read-gated at the proxy layer, so a GET here only
 * fires for an already-authenticated reader).
 */
import { randomUUID } from "node:crypto";

import { requireSessionAuth } from "@/lib/auth";
import {
  appendComment,
  commentsPathForSlug,
  isValidSlug,
  readComments,
  rewriteComments,
  withFileLock,
  type DocCommentRow,
} from "@/lib/doc-comments";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_BODY_CHARS = 10_000;
const MAX_QUOTE_CHARS = 2_000;
const MAX_LABEL_CHARS = 300;
const MAX_ID_CHARS = 200;

function clip(s: unknown, max: number): string | undefined {
  if (typeof s !== "string") return undefined;
  const t = s.trim();
  if (!t) return undefined;
  return t.length > max ? t.slice(0, max) : t;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const slug = searchParams.get("slug") ?? "";
  const file = commentsPathForSlug(slug);
  if (!file) return Response.json({ ok: false, error: "invalid slug" }, { status: 400 });
  const comments = await readComments(file);
  return Response.json({ ok: true, comments });
}

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

  const slug = typeof obj.slug === "string" ? obj.slug : "";
  const file = commentsPathForSlug(slug);
  if (!file || !isValidSlug(slug)) {
    return Response.json({ ok: false, error: "invalid slug" }, { status: 400 });
  }

  const body = clip(obj.body, MAX_BODY_CHARS);
  if (!body) {
    return Response.json({ ok: false, error: "empty body" }, { status: 400 });
  }

  const row: DocCommentRow = {
    id: `dc-${randomUUID()}`,
    ts: new Date().toISOString(),
    author: user.email,
    kind: "doc-comment",
    body,
  };
  const sectionId = clip(obj.section_id, MAX_LABEL_CHARS);
  const sectionLabel = clip(obj.section_label, MAX_LABEL_CHARS);
  const quote = clip(obj.quote, MAX_QUOTE_CHARS);
  const inReplyTo = clip(obj.in_reply_to, MAX_ID_CHARS);
  if (sectionId) row.section_id = sectionId;
  if (sectionLabel) row.section_label = sectionLabel;
  if (quote) row.quote = quote;
  if (inReplyTo) {
    row.in_reply_to = inReplyTo;
    row.kind = "doc-comment-reply";
  }

  await withFileLock(file, async () => {
    await appendComment(file, row);
  });

  return Response.json({ ok: true, id: row.id });
}

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
  const slug = typeof obj.slug === "string" ? obj.slug : "";
  const commentId = typeof obj.commentId === "string" ? obj.commentId : "";
  const file = commentsPathForSlug(slug);
  if (!file || !isValidSlug(slug) || !commentId) {
    return Response.json({ ok: false, error: "invalid request" }, { status: 400 });
  }

  let removed = false;
  await withFileLock(file, async () => {
    const rows = await readComments(file);
    const next = rows.filter((r) => {
      const drop = r.id === commentId && r.author === user.email;
      if (drop) removed = true;
      return !drop;
    });
    if (removed) await rewriteComments(file, next);
  });

  return Response.json({ ok: removed });
}
