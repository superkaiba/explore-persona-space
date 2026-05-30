"use client";

/**
 * TaskCommentBody — the client shell for a task body + its anchored
 * comments. Mirrors DocBody (the /docs/<slug> shell) but binds to the
 * TASK comment API (`/api/updates/comment`) instead of the docs API.
 *
 * Shape differences from DocBody / the docs API:
 *   - The anchor is NESTED: `anchor.quote` (NOT a top-level `quote`).
 *   - The id field is `taskId` (number), NOT `slug`.
 *   - Comment kinds are `anchor-comment` / `anchor-comment-reply`.
 *   - The suppress flag is `archived` (NOT `addressed`); there is no
 *     "Address all" affordance for tasks.
 *
 * Wiring:
 *   - <AnchoredCommentsProvider anchors={…}> supplies the committed
 *     comment anchors (open, non-archived `anchor-comment` rows whose
 *     `anchor.quote` is ≥4 chars) so MarkdownDoc wraps each in <mark> on
 *     the rendered body — the same highlight-to-comment surface docs use.
 *   - <MarkdownDoc showToc enableCollapsibleSections docId={taskId}> renders
 *     the body. Comment writes + Ask-Claude are enabled only for signed-in
 *     editors on a non-readOnly mount; `public` is passed when readOnly OR
 *     not editor-authed, which disables the selection popover + Ask-Claude.
 *   - <TaskCommentRail> reads the pending selection from context, POSTs to
 *     `/api/updates/comment`, lists open comments (nesting reply rows under
 *     their parent), supports delete for the author's own rows, hover→mark
 *     sync, and click-quote→scroll. `@claude` in a comment triggers an async
 *     reply row, surfaced by polling after submit (same as CardCommentBox).
 *
 * The selection → popover → pendingQuote → rail composer path is the
 * highlight flow; the rail mirrors the DocCommentRail composer.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import {
  AnchoredCommentsProvider,
  useAnchoredComments,
  type AnchorRecord,
} from "@/app/tasks/[id]/AnchoredCommentsContext";
import { MarkdownDoc } from "@/components/MarkdownDoc";

/** A task comment row as returned by GET /api/updates/comment?taskId=N. */
export type TaskCommentView = {
  id: string;
  ts: string;
  author: string;
  kind: "anchor-comment" | "anchor-comment-reply";
  body: string;
  anchor?: { quote: string; prefix?: string; suffix?: string };
  in_reply_to?: string;
  archived?: boolean;
};

const PENDING_REPLY_TIMEOUT_MS = 60_000;

/** Client placeholder shown while a `@claude` reply is being generated. */
type PendingPlaceholder = {
  id: string;
  parentId: string;
  startedAt: number;
  state: "pending" | "error";
};

export function TaskCommentBody({
  taskId,
  body,
  title,
  isLegacyHtml,
  initialComments,
  editorAuthed,
  currentUserEmail,
  readOnly = false,
}: {
  taskId: number;
  body: string;
  title: string;
  isLegacyHtml: boolean;
  initialComments: TaskCommentView[];
  editorAuthed: boolean;
  currentUserEmail: string | null;
  readOnly?: boolean;
}) {
  const [comments, setComments] = useState<TaskCommentView[]>(initialComments);

  // Writes are allowed only for a signed-in editor on a non-readOnly mount.
  const canWrite = editorAuthed && !readOnly;
  // MarkdownDoc's `public` flag gates the selection popover + Ask-Claude.
  const isPublic = readOnly || !editorAuthed;

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`/api/updates/comment?taskId=${taskId}`, {
        cache: "no-store",
        credentials: "same-origin",
      });
      if (res.status === 401) {
        // Unauthenticated viewer — keep whatever the server passed in.
        return;
      }
      const data = (await res.json()) as
        | { ok: true; comments: TaskCommentView[] }
        | { ok: false; error: string };
      if (data.ok) setComments(data.comments);
    } catch {
      /* leave existing list; transient fetch error */
    }
  }, [taskId]);

  // Archived closure: an archived anchor-comment hides its whole subtree
  // (synthesis reply + any user follow-ups). Iterative because a thread
  // can be 3+ deep; 64-hop cap is a sanity bound. Mirrors CommentsList
  // and CardCommentBox.
  const archivedIds = useMemo(() => {
    const closed = new Set<string>();
    for (const c of comments) {
      if (c.kind === "anchor-comment" && c.archived === true) closed.add(c.id);
    }
    let changed = true;
    let hops = 0;
    while (changed && hops < 64) {
      changed = false;
      for (const c of comments) {
        if (c.in_reply_to && closed.has(c.in_reply_to) && !closed.has(c.id)) {
          closed.add(c.id);
          changed = true;
        }
      }
      hops++;
    }
    return closed;
  }, [comments]);

  const visibleComments = useMemo(
    () => comments.filter((c) => !archivedIds.has(c.id)),
    [comments, archivedIds],
  );

  // Committed anchors: open root anchor-comments carrying a `anchor.quote`
  // ≥4 chars. MarkdownDoc wraps each occurrence in <mark data-comment-id>.
  // Reply rows share the parent's anchor — don't double-wrap.
  const anchors: AnchorRecord[] = useMemo(
    () =>
      visibleComments
        .filter(
          (c) =>
            c.kind === "anchor-comment" &&
            typeof c.anchor?.quote === "string" &&
            c.anchor.quote.trim().length >= 4,
        )
        .map((c) => ({ id: c.id, quote: (c.anchor!.quote as string).trim() })),
    [visibleComments],
  );

  // Inline-composer create hook: POST the anchored comment + refetch. Wired
  // into MarkdownDoc so highlight-to-comment opens the inline composer at the
  // selection (works at any width) instead of relying on the side rail. The
  // rail stays as the comment LIST + whole-task composer.
  const onCommentCreate = useCallback(
    async ({ quote, body: text }: { quote: string; body: string }): Promise<boolean> => {
      if (!canWrite) return false;
      const trimmed = text.trim();
      if (!trimmed) return false;
      try {
        const payload: { taskId: number; body: string; anchor?: { quote: string } } = {
          taskId,
          body: trimmed,
        };
        const q = quote.trim();
        if (q) payload.anchor = { quote: q };
        const res = await fetch("/api/updates/comment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify(payload),
        });
        const data = (await res.json()) as
          | { ok: true; will_reply?: boolean }
          | { ok: false; error: string };
        if (!data.ok) return false;
        await refresh();
        if (data.will_reply) {
          for (const delayMs of [3_000, 8_000, 20_000, 45_000]) {
            setTimeout(() => void refresh(), delayMs);
          }
        }
        return true;
      } catch {
        return false;
      }
    },
    [taskId, canWrite, refresh],
  );

  return (
    <AnchoredCommentsProvider
      anchors={anchors}
      onCommentCreate={canWrite ? onCommentCreate : null}
    >
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <MarkdownDoc
            body={body}
            isLegacyHtml={isLegacyHtml}
            showToc
            enableCollapsibleSections
            docId={taskId}
            enableAskClaude={canWrite}
            askClaudeTitle={title}
            public={isPublic}
            onCommentCreate={canWrite ? onCommentCreate : undefined}
          />
        </div>
        <aside className="lg:sticky lg:top-4 lg:self-start">
          <TaskCommentRail
            taskId={taskId}
            comments={visibleComments}
            canWrite={canWrite}
            currentUserEmail={currentUserEmail}
            signInNext={`/results/${taskId}`}
            onChanged={refresh}
          />
        </aside>
      </div>
    </AnchoredCommentsProvider>
  );
}

function TaskCommentRail({
  taskId,
  comments,
  canWrite,
  currentUserEmail,
  signInNext,
  onChanged,
}: {
  taskId: number;
  comments: TaskCommentView[];
  canWrite: boolean;
  currentUserEmail: string | null;
  signInNext: string;
  onChanged: () => void | Promise<void>;
}) {
  const { pendingQuote, setPendingQuote, setHoveredId, requestScrollTo } =
    useAnchoredComments();
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingPlaceholder[]>([]);
  const pendingTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map(),
  );

  // Clear any pending placeholder whose real reply row has landed.
  useEffect(() => {
    const landedIds = new Set(comments.map((c) => c.id));
    setPending((prev) => {
      let changed = false;
      const next = prev.filter((p) => {
        if (landedIds.has(p.id)) {
          const t = pendingTimersRef.current.get(p.id);
          if (t) {
            clearTimeout(t);
            pendingTimersRef.current.delete(p.id);
          }
          changed = true;
          return false;
        }
        return true;
      });
      return changed ? next : prev;
    });
  }, [comments]);

  useEffect(() => {
    const timers = pendingTimersRef.current;
    return () => {
      for (const t of timers.values()) clearTimeout(t);
      timers.clear();
    };
  }, []);

  // Root anchor-comments are the rail entries; replies nest beneath them.
  const roots = useMemo(
    () => comments.filter((c) => c.kind === "anchor-comment" && !c.in_reply_to),
    [comments],
  );
  const repliesByParent = useMemo(() => {
    const m = new Map<string, TaskCommentView[]>();
    for (const c of comments) {
      if (!c.in_reply_to) continue;
      const arr = m.get(c.in_reply_to) ?? [];
      arr.push(c);
      m.set(c.in_reply_to, arr);
    }
    for (const arr of m.values()) {
      arr.sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
    }
    return m;
  }, [comments]);

  const onPendingStart = useCallback((parentId: string, replyId: string) => {
    setPending((prev) => {
      if (prev.some((p) => p.id === replyId)) return prev;
      return [
        ...prev,
        { id: replyId, parentId, startedAt: Date.now(), state: "pending" as const },
      ];
    });
    const timer = setTimeout(() => {
      setPending((prev) =>
        prev.map((p) => (p.id === replyId ? { ...p, state: "error" as const } : p)),
      );
    }, PENDING_REPLY_TIMEOUT_MS);
    pendingTimersRef.current.set(replyId, timer);
  }, []);

  async function submit() {
    if (!draft.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const payload: {
        taskId: number;
        body: string;
        anchor?: { quote: string };
      } = { taskId, body: draft.trim() };
      if (pendingQuote?.trim()) payload.anchor = { quote: pendingQuote.trim() };
      const res = await fetch("/api/updates/comment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });
      const data = (await res.json()) as
        | {
            ok: true;
            id: string;
            ts?: string;
            will_reply?: boolean;
            pending_reply_id?: string;
          }
        | { ok: false; error: string };
      if (!data.ok) {
        setError(
          data.error === "unauthorized" ? "Sign in to comment." : data.error || "failed",
        );
        return;
      }
      setDraft("");
      setPendingQuote(null);
      if (data.will_reply && data.pending_reply_id) {
        onPendingStart(data.id, data.pending_reply_id);
      }
      await onChanged();
      // Claude's auto-reply (when `@claude` was used) lands a few seconds
      // later via the server's fire-and-forget. Poll a few times so it
      // surfaces without a manual refresh.
      if (data.will_reply) {
        for (const delayMs of [3_000, 8_000, 20_000, 45_000]) {
          setTimeout(() => {
            void onChanged();
          }, delayMs);
        }
      }
    } catch {
      setError("network error");
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    try {
      await fetch("/api/updates/comment", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ taskId, commentId: id }),
      });
      await onChanged();
    } catch {
      /* ignore; next refresh reconciles */
    }
  }

  return (
    <section className="rounded-lg border border-stone-200 bg-white p-4 text-sm">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
          Comments{" "}
          {roots.length > 0 && (
            <span className="text-stone-400">({roots.length})</span>
          )}
        </h2>
      </div>

      {canWrite ? (
        <div className="mt-3 space-y-2">
          {pendingQuote ? (
            <div className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-2 py-1.5 text-xs">
              <div className="flex-1">
                <div className="font-medium text-amber-900">Commenting on selection:</div>
                <blockquote className="mt-0.5 line-clamp-3 italic text-amber-950">
                  &ldquo;
                  {pendingQuote.length > 200 ? pendingQuote.slice(0, 200) + "…" : pendingQuote}
                  &rdquo;
                </blockquote>
              </div>
              <button
                type="button"
                onClick={() => setPendingQuote(null)}
                className="text-amber-700 hover:text-amber-900"
                aria-label="Clear anchor"
                title="Clear anchor"
              >
                ✕
              </button>
            </div>
          ) : (
            <p className="text-[11px] text-stone-400">
              Select text in the body to anchor a comment, or comment on the whole task. Mention{" "}
              <code className="rounded bg-stone-100 px-1">@claude</code> to summon a reply.
            </p>
          )}
          <textarea
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
                e.preventDefault();
                void submit();
              }
            }}
            placeholder={
              pendingQuote ? "Comment on this selection…" : "Leave a comment on this task…"
            }
            rows={3}
            className="w-full rounded border border-stone-300 px-2 py-1.5 text-sm text-stone-800 placeholder:text-stone-400"
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-rose-600">{error}</span>
            <button
              type="button"
              onClick={() => void submit()}
              disabled={!draft.trim() || busy}
              className="rounded border border-stone-300 px-3 py-1 text-xs font-medium text-stone-700 transition-colors hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? "Saving…" : pendingQuote ? "Post anchored comment" : "Comment"}
            </button>
          </div>
        </div>
      ) : (
        <p className="mt-3 rounded border border-dashed border-stone-300 px-3 py-2 text-xs text-stone-500">
          <Link
            href={`/sign-in?next=${encodeURIComponent(signInNext)}`}
            className="font-medium underline"
          >
            Sign in
          </Link>{" "}
          to comment.
        </p>
      )}

      {roots.length > 0 && (
        <ul className="mt-4 space-y-3 border-t border-stone-100 pt-3">
          {roots.map((c) => (
            <li
              key={c.id}
              className="space-y-1"
              onMouseEnter={() => c.anchor?.quote && setHoveredId(c.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              {c.anchor?.quote && (
                <button
                  type="button"
                  onClick={() => requestScrollTo(c.id)}
                  className="block w-full border-l-2 border-amber-300 pl-2 text-left text-xs italic text-stone-500 hover:text-stone-700"
                  title="Scroll to highlighted text"
                >
                  &ldquo;
                  {c.anchor.quote.length > 160
                    ? c.anchor.quote.slice(0, 160) + "…"
                    : c.anchor.quote}
                  &rdquo;
                </button>
              )}
              <CommentBody body={c.body} />
              <div className="flex items-center gap-2 text-[11px] text-stone-400">
                <span>{c.author}</span>
                <time className="tabular-nums">{compactTs(c.ts)}</time>
                {currentUserEmail && c.author === currentUserEmail && (
                  <button
                    type="button"
                    onClick={() => remove(c.id)}
                    className="hover:text-rose-600"
                  >
                    delete
                  </button>
                )}
              </div>
              <ThreadReplies
                parentId={c.id}
                repliesByParent={repliesByParent}
                currentUserEmail={currentUserEmail}
                onDelete={remove}
              />
              <PendingForParent parentId={c.id} pending={pending} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/**
 * Render the reply subtree under a parent comment. Claude replies and
 * user follow-ups both flow through here; recurse so 3+-deep threads
 * render. Mirrors the nesting in CardCommentBox's ArchivedReplies but for
 * the live (non-archived) rail.
 */
function ThreadReplies({
  parentId,
  repliesByParent,
  currentUserEmail,
  onDelete,
}: {
  parentId: string;
  repliesByParent: Map<string, TaskCommentView[]>;
  currentUserEmail: string | null;
  onDelete: (id: string) => void;
}) {
  const replies = repliesByParent.get(parentId);
  if (!replies || replies.length === 0) return null;
  return (
    <ul className="mt-1.5 space-y-1.5 border-l-2 border-stone-200 pl-3">
      {replies.map((c) => {
        const isClaude = c.author === "claude" || c.kind === "anchor-comment-reply";
        return (
          <li key={c.id} className="space-y-1">
            <CommentBody body={c.body} />
            <div className="flex items-center gap-2 text-[11px] text-stone-400">
              <span className="font-medium text-stone-600">
                {isClaude ? "Claude" : c.author}
              </span>
              <time className="tabular-nums">{compactTs(c.ts)}</time>
              {currentUserEmail && c.author === currentUserEmail && (
                <button
                  type="button"
                  onClick={() => onDelete(c.id)}
                  className="hover:text-rose-600"
                >
                  delete
                </button>
              )}
            </div>
            <ThreadReplies
              parentId={c.id}
              repliesByParent={repliesByParent}
              currentUserEmail={currentUserEmail}
              onDelete={onDelete}
            />
          </li>
        );
      })}
    </ul>
  );
}

function PendingForParent({
  parentId,
  pending,
}: {
  parentId: string;
  pending: PendingPlaceholder[];
}) {
  const mine = pending.filter((p) => p.parentId === parentId);
  if (mine.length === 0) return null;
  return (
    <ul className="mt-1.5 space-y-1.5 border-l-2 border-stone-200 pl-3">
      {mine.map((p) => (
        <li
          key={p.id}
          className="text-[11px] italic text-stone-400"
        >
          {p.state === "pending"
            ? "Claude is thinking…"
            : "Claude didn't reply in time (it may still arrive on refresh)."}
        </li>
      ))}
    </ul>
  );
}

function CommentBody({ body }: { body: string }) {
  return (
    <div className="prose prose-sm prose-stone max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeHighlight]}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
}

function compactTs(ts: string): string {
  const m = ts.match(/^\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]}` : ts;
}
