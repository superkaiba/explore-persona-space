"use client";

/**
 * CommentRail — the right-hand, vertically-aligned anchored-comment rail for
 * /tasks/[id] (the Sagan / Google-Docs margin pattern the /updates cards use
 * via CardCommentBox + CommentList).
 *
 * Each anchored comment is absolutely positioned by <CommentList> at the Y of
 * its <mark> in the body (positions published through AnchoredCommentsContext
 * by every <MarkdownDoc> on the page), so comments line up beside the text they
 * annotate and stack into one vertical column. Unanchored / whole-task comments
 * flow at the bottom; a compose box sits below the list.
 *
 * Shown at `lg+` only (the absolute alignment math assumes the rail shares its
 * top with the feed column). On narrower viewports the feed falls back to the
 * stacked <TaskCommentsPanel> above the body — see TaskFeed.
 */
import { useCallback, useState } from "react";
import Link from "next/link";
import { CommentList } from "@/app/tasks/[id]/CommentList";
import type { TaskCommentView } from "@/app/tasks/[id]/TaskCommentBody";
import type { TaskComment } from "@/lib/tasks";

export function CommentRail({
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
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <aside className="hidden min-w-0 lg:block">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-stone-400">
        Comments
      </div>
      <CommentList
        comments={comments as unknown as TaskComment[]}
        onDelete={canWrite ? (id) => void remove(id) : undefined}
        reply={
          canWrite && currentUserEmail
            ? {
                taskId,
                currentUserEmail,
                onPosted: async () => {
                  await onChanged();
                },
                onPendingStart: () => {},
              }
            : undefined
        }
      />
      {canWrite ? (
        <div className="mt-4 space-y-2 border-t border-stone-100 pt-3">
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
            placeholder="Comment on this task… (or highlight text on a card)"
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
        <p className="mt-3 rounded border border-dashed border-stone-300 px-3 py-2 text-xs text-stone-500">
          <Link
            href={`/sign-in?next=${encodeURIComponent(`/results/${taskId}`)}`}
            className="font-medium underline"
          >
            Sign in
          </Link>{" "}
          to comment.
        </p>
      )}
    </aside>
  );
}
