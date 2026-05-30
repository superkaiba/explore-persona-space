"use client";

/**
 * TaskCommentsPanel — the page-level comment LIST + manage surface for
 * /tasks/[id].
 *
 * This is the reliably-visible home for viewing / deleting comments and
 * leaving a whole-task (un-anchored) comment. It replaces the side rail that
 * used to live inside the body card and stacked ~22k px below the body on
 * narrow viewports. It renders ABOVE the feed as a collapsible "Comments (N)"
 * panel so it's never buried.
 *
 * Anchored comments are CREATED inline at the selection (see MarkdownDoc's
 * inline composer, wired via context in <TaskFeed>); this panel is for
 * viewing/managing them + posting non-anchored task-level comments. Hovering a
 * row syncs the matching body <mark>; clicking the quote scrolls to it.
 *
 * Thread rendering (Claude replies + user follow-ups) mirrors the old
 * TaskCommentRail's nesting.
 */
import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useAnchoredComments } from "@/app/tasks/[id]/AnchoredCommentsContext";
import type { TaskCommentView } from "@/app/tasks/[id]/TaskCommentBody";

export function TaskCommentsPanel({
  taskId,
  comments,
  canWrite,
  currentUserEmail,
  onChanged,
}: {
  taskId: number;
  comments: TaskCommentView[];
  canWrite: boolean;
  currentUserEmail: string | null;
  onChanged: () => void | Promise<void>;
}) {
  const { setHoveredId, requestScrollTo } = useAnchoredComments();
  const [open, setOpen] = useState(true);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const submit = useCallback(async () => {
    const text = draft.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/updates/comment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ taskId, body: text }),
      });
      const data = (await res.json()) as
        | { ok: true; will_reply?: boolean }
        | { ok: false; error: string };
      if (!data.ok) {
        setError(
          data.error === "unauthorized" ? "Sign in to comment." : data.error || "failed",
        );
        return;
      }
      setDraft("");
      await onChanged();
      if (data.will_reply) {
        for (const delayMs of [3_000, 8_000, 20_000, 45_000]) {
          setTimeout(() => void onChanged(), delayMs);
        }
      }
    } catch {
      setError("network error");
    } finally {
      setBusy(false);
    }
  }, [draft, busy, taskId, onChanged]);

  const remove = useCallback(
    async (id: string) => {
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
    },
    [taskId, onChanged],
  );

  return (
    <section className="rounded-lg border border-stone-200 bg-white text-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-stone-500 hover:bg-stone-50 sm:px-6"
      >
        <span className="text-stone-400" aria-hidden>
          {open ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </span>
        Comments {roots.length > 0 && <span className="text-stone-400">({roots.length})</span>}
        <span className="ml-auto font-normal normal-case text-[11px] text-stone-400">
          select text on any card to comment
        </span>
      </button>

      {open && (
        <div className="border-t border-stone-100 px-4 pb-4 pt-3 sm:px-6">
          {canWrite ? (
            <div className="space-y-2">
              <p className="text-[11px] text-stone-400">
                Highlight text on any card (body, plan, or an event) to anchor a
                comment, or leave a whole-task comment here. Mention{" "}
                <code className="rounded bg-stone-100 px-1">@claude</code> to summon a
                reply.
              </p>
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
                placeholder="Leave a comment on this task…"
                rows={2}
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
                  {busy ? "Saving…" : "Comment"}
                </button>
              </div>
            </div>
          ) : (
            <p className="rounded border border-dashed border-stone-300 px-3 py-2 text-xs text-stone-500">
              <Link
                href={`/sign-in?next=${encodeURIComponent(`/results/${taskId}`)}`}
                className="font-medium underline"
              >
                Sign in
              </Link>{" "}
              to comment.
            </p>
          )}

          {roots.length > 0 ? (
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
                  <CommentMarkdown body={c.body} />
                  <div className="flex items-center gap-2 text-[11px] text-stone-400">
                    <span>{c.author}</span>
                    <time className="tabular-nums">{compactTs(c.ts)}</time>
                    {currentUserEmail && c.author === currentUserEmail && (
                      <button
                        type="button"
                        onClick={() => void remove(c.id)}
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
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-[11px] text-stone-400">No comments yet.</p>
          )}
        </div>
      )}
    </section>
  );
}

function ThreadReplies({
  parentId,
  repliesByParent,
  currentUserEmail,
  onDelete,
}: {
  parentId: string;
  repliesByParent: Map<string, TaskCommentView[]>;
  currentUserEmail: string | null;
  onDelete: (id: string) => void | Promise<void>;
}) {
  const replies = repliesByParent.get(parentId);
  if (!replies || replies.length === 0) return null;
  return (
    <ul className="mt-1.5 space-y-1.5 border-l-2 border-stone-200 pl-3">
      {replies.map((c) => {
        const isClaude = c.author === "claude" || c.kind === "anchor-comment-reply";
        return (
          <li key={c.id} className="space-y-1">
            <CommentMarkdown body={c.body} />
            <div className="flex items-center gap-2 text-[11px] text-stone-400">
              <span className="font-medium text-stone-600">
                {isClaude ? "Claude" : c.author}
              </span>
              <time className="tabular-nums">{compactTs(c.ts)}</time>
              {currentUserEmail && c.author === currentUserEmail && (
                <button
                  type="button"
                  onClick={() => void onDelete(c.id)}
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

function CommentMarkdown({ body }: { body: string }) {
  return (
    <div className="prose prose-sm prose-stone max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, rehypeHighlight]}>
        {body}
      </ReactMarkdown>
    </div>
  );
}

function compactTs(ts: string): string {
  const m = ts.match(/^\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]}` : ts;
}
