/**
 * POST /api/docs/address-comments — the "Address all" button on a
 * /docs/<slug> page.
 *
 * Loads the raw doc (resolved via the shared docs path resolver — works for
 * top-level docs AND virtual mentor-update / activity / idea slugs) + every
 * open `doc-comment`, asks Claude (via the local sidecar) to rewrite the doc
 * to address them, then:
 *   1. writes the revised doc to disk (the /docs route is force-dynamic, so
 *      it shows immediately — no rebuild),
 *   2. rewrites the doc's gitignored `.comments/<stem>.jsonl` to mark each
 *      addressed row with `addressed: true`, `addressed_in: <head sha>`,
 *      `addressed_note`,
 *   3. appends one synthesis `doc-comment-reply` row.
 *
 * Unlike the code-edit comment path, Claude never touches the filesystem
 * here — it returns text, and THIS route is the only writer, and only to
 * the validated doc path. Auth: `isEditorAuthed()`-gated.
 */
import { randomUUID } from "node:crypto";

import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { checkRateLimit, clientKey } from "@/lib/rate-limit";
import { parseAddressJson, readHeadSha, streamSidecarChat } from "@/lib/claude-comment-ops";
import {
  commentsPathForSlug,
  isValidSlug,
  readComments,
  readDocRaw,
  rewriteComments,
  withFileLock,
  writeDocRaw,
  type DocCommentRow,
} from "@/lib/doc-comments";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_DOC_CHARS = 1_000_000;
const REPLY_TIMEOUT_MS = 5 * 60 * 1000;

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + "…" : s;
}

export async function POST(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });
  if (!(await isEditorAuthed())) {
    return Response.json({ ok: false, error: "editor cookie required" }, { status: 403 });
  }

  const rl = checkRateLimit("sidecar-chat", clientKey(request));
  if (!rl.allowed) {
    return Response.json(
      { ok: false, error: "rate limited", retryAfterS: rl.retryAfterS },
      { status: 429 },
    );
  }

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

  const currentDoc = await readDocRaw(slug);
  if (currentDoc === null) {
    return Response.json({ ok: false, error: "doc not found" }, { status: 404 });
  }
  if (currentDoc.length > MAX_DOC_CHARS) {
    return Response.json({ ok: false, error: "doc too large" }, { status: 413 });
  }

  const allRows = await readComments(file);
  const open = allRows.filter((r) => r.kind === "doc-comment" && r.addressed !== true && r.body);
  if (open.length === 0) {
    return Response.json({ ok: false, error: "no open comments to address" }, { status: 409 });
  }

  const prompt = [
    `You are revising the markdown research document "${slug}.md" to address the reviewer comments below.`,
    "Preserve the document's structure, the YAML frontmatter (if any), the HTML-comment anchors",
    "(e.g. <!-- q:... -->), and the Belief/Confidence/Evidence line format. Only change what the",
    "comments ask for, plus minimal edits needed for consistency.",
    "Return the FULL revised markdown document, then on a new line a single JSON object",
    'of the form: {"addressed": {"<comment_id>": "<one-sentence note on how you addressed it>", ...}}.',
    "Include every comment_id you actually addressed. No commentary outside those two parts.",
    "No code fences around the whole document.",
    "",
    "Current document:",
    currentDoc,
    "",
    "Open comments:",
    ...open.map((c) => {
      const where = c.section_label ? `\n  about: ${truncate(c.section_label, 200)}` : "";
      const quote = c.quote ? `\n  quote: ${truncate(c.quote, 400)}` : "";
      return `- id: ${c.id}${where}${quote}\n  comment: ${truncate(c.body, 1200)}`;
    }),
  ].join("\n");

  const raw = await streamSidecarChat({
    sessionId: `docs-address-${slug}`,
    prompt,
    timeoutMs: REPLY_TIMEOUT_MS,
    maxChars: 1_200_000,
  });
  if (!raw) {
    return Response.json({ ok: false, error: "sidecar returned no content" }, { status: 502 });
  }
  const parsed = parseAddressJson(raw);
  if (!parsed) {
    return Response.json(
      { ok: false, error: "could not parse revised doc from sidecar response" },
      { status: 502 },
    );
  }
  const { body: newDoc, addressed: addressedMap } = parsed;
  if (!newDoc.trim()) {
    return Response.json({ ok: false, error: "revised doc was empty" }, { status: 502 });
  }

  const wrote = await writeDocRaw(slug, newDoc);
  if (!wrote) {
    return Response.json({ ok: false, error: "failed to write doc" }, { status: 500 });
  }
  const sha = await readHeadSha();

  const knownIds = new Set(open.map((c) => c.id));
  const appliedNotes: Array<{ id: string; note: string }> = [];
  for (const [id, note] of Object.entries(addressedMap)) {
    if (knownIds.has(id) && typeof note === "string") appliedNotes.push({ id, note });
  }

  await withFileLock(file, async () => {
    const rows = await readComments(file);
    const noteById = new Map(appliedNotes.map((x) => [x.id, x.note]));
    for (const row of rows) {
      if (noteById.has(row.id) && row.kind === "doc-comment") {
        row.addressed = true;
        row.addressed_in = sha;
        row.addressed_note = noteById.get(row.id);
      }
    }
    if (appliedNotes.length > 0) {
      const synthesis: DocCommentRow = {
        id: `dcr-${randomUUID()}`,
        ts: new Date().toISOString(),
        author: "claude",
        kind: "doc-comment-reply",
        body: buildSynthesis(appliedNotes, sha),
        in_reply_to: appliedNotes[0].id,
      };
      rows.push(synthesis);
    }
    await rewriteComments(file, rows);
  });

  return Response.json({ ok: true, addressed: appliedNotes.map((n) => n.id), sha });
}

function buildSynthesis(notes: Array<{ id: string; note: string }>, sha: string): string {
  const list = notes
    .slice(0, 20)
    .map((n) => `- \`${n.id}\`: ${n.note}`)
    .join("\n");
  return [
    `Addressed ${notes.length} comment${notes.length === 1 ? "" : "s"} (head \`${sha}\`):`,
    "",
    list,
  ].join("\n");
}
