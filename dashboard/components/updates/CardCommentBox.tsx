"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AnchoredCommentsProvider,
  useAnchoredComments,
} from "@/app/tasks/[id]/AnchoredCommentsContext";
import { CommentableBody } from "@/app/tasks/[id]/CommentableBody";
import { CommentList } from "@/app/tasks/[id]/CommentList";
import type { TaskComment } from "@/lib/tasks";

/**
 * Per-card anchored comments wrapper, used on the /updates cards and
 * inside the modal full-view. Owns: client-side comment-fetch + refresh,
 * the inline composer that reads `pendingQuote` from the
 * AnchoredCommentsProvider, the POST to /api/updates/comment, and the
 * DELETE handler.
 *
 * Two layouts:
 *   - `layout="inline"` (default): comments stack below the body,
 *     each with its own header/markdown/delete button.
 *   - `layout="rail"`: comments render in a sidebar to the right of the
 *     body, vertically aligned with their <mark> via the existing
 *     CommentList margin-math (mirrors /tasks/[id] page layout).
 *
 * `taskId` is the GitHub issue number (= `tasks/<N>/`). Cards without
 * a `githubIssueNumber` cannot host anchored comments and should not
 * render this wrapper.
 */

type Layout = "inline" | "rail";

type FetchedComment = TaskComment & {
  anchor?: { quote: string; prefix?: string; suffix?: string };
  parent_id?: string;
};

export function CardCommentBox({
  taskId,
  body,
  currentUserEmail,
  layout = "inline",
}: {
  taskId: number;
  body: string;
  currentUserEmail: string | null;
  layout?: Layout;
}) {
  const [comments, setComments] = useState<FetchedComment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/updates/comment?taskId=${taskId}`,
        { cache: "no-store", credentials: "same-origin" },
      );
      if (res.status === 401) {
        // Unauthenticated viewer — show body, no comments UI.
        setComments([]);
        setError(null);
        setLoading(false);
        return;
      }
      const json = (await res.json()) as
        | { ok: true; comments: FetchedComment[] }
        | { ok: false; error: string };
      if (!json.ok) {
        setError(json.error);
        setLoading(false);
        return;
      }
      setComments(json.comments);
      setError(null);
      setLoading(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const anchors = useMemo(
    () =>
      comments
        // Reply rows share their parent's anchor; don't double-wrap. Only
        // user-authored anchor-comment rows contribute a <mark>.
        .filter(
          (c) =>
            c.kind === "anchor-comment" &&
            typeof c.anchor?.quote === "string" &&
            c.anchor!.quote.length > 0,
        )
        .map((c) => ({ id: c.id, quote: c.anchor!.quote })),
    [comments],
  );

  return (
    <AnchoredCommentsProvider anchors={anchors}>
      <CardCommentBoxInner
        taskId={taskId}
        body={body}
        comments={comments as TaskComment[]}
        currentUserEmail={currentUserEmail}
        layout={layout}
        loading={loading}
        error={error}
        onRefresh={refresh}
      />
    </AnchoredCommentsProvider>
  );
}

function CardCommentBoxInner({
  taskId,
  body,
  comments,
  currentUserEmail,
  layout,
  loading,
  error,
  onRefresh,
}: {
  taskId: number;
  body: string;
  comments: TaskComment[];
  currentUserEmail: string | null;
  layout: Layout;
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<void>;
}) {
  const handleDelete = useCallback(
    async (commentId: string) => {
      try {
        const res = await fetch("/api/updates/comment", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({ taskId, commentId }),
        });
        if (!res.ok) {
          const json = (await res.json().catch(() => ({}))) as { error?: string };
          window.alert(`Delete failed: ${json.error ?? res.statusText}`);
          return;
        }
        await onRefresh();
      } catch (e) {
        window.alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
    [taskId, onRefresh],
  );

  // The CommentList delete control is wired per-row, but the server
  // also enforces author-match. Show the button on every row when the
  // viewer is signed in; the server returns 403 if they aren't the
  // author of that particular row.
  const onDelete = currentUserEmail ? handleDelete : undefined;

  if (layout === "rail") {
    return (
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <CommentableBody body={body} isLegacyHtml={false} />
        </div>
        <aside className="min-w-0">
          <CardComposer
            taskId={taskId}
            currentUserEmail={currentUserEmail}
            onPosted={onRefresh}
          />
          <div className="mt-4">
            {loading ? null : error ? (
              <CommentsError error={error} />
            ) : (
              <CommentList comments={comments} inline onDelete={onDelete} />
            )}
          </div>
        </aside>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <CommentableBody body={body} isLegacyHtml={false} />
      <CardComposer
        taskId={taskId}
        currentUserEmail={currentUserEmail}
        onPosted={onRefresh}
      />
      {loading ? null : error ? (
        <CommentsError error={error} />
      ) : (
        <CommentList comments={comments} inline onDelete={onDelete} />
      )}
    </div>
  );
}

function CommentsError({ error }: { error: string }) {
  return (
    <p className="rounded border border-dashed border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700">
      Could not load comments: {error}
    </p>
  );
}

/**
 * Inline composer wired to AnchoredCommentsProvider. Watches
 * `pendingQuote` from the provider, attaches `{quote, prefix, suffix}`
 * if present, POSTs to /api/updates/comment, then clears the pending
 * mark + body so the next selection starts fresh.
 *
 * Prefix/suffix are computed from the body text neighborhood — we look
 * up the FIRST occurrence of `quote` in the markdown source and slice
 * 80 chars of context on each side. The /tasks/[id] flow uses these
 * for disambiguation when the same quote appears multiple times; on
 * /updates cards the body is short enough that the first occurrence
 * is usually the only one, but we include them for shape parity.
 */
function CardComposer({
  taskId,
  currentUserEmail,
  onPosted,
}: {
  taskId: number;
  currentUserEmail: string | null;
  onPosted: () => Promise<void>;
}) {
  const { pendingQuote, setPendingQuote } = useAnchoredComments();
  const [draft, setDraft] = useState("");
  const [posting, setPosting] = useState(false);
  const [status, setStatus] = useState<
    { kind: "ok" | "err"; text: string } | null
  >(null);

  if (!currentUserEmail) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-3 py-2 text-xs text-stone-500">
        Sign in to comment on this result.
      </p>
    );
  }

  async function onSubmit() {
    if (!draft.trim() || posting) return;
    setPosting(true);
    setStatus(null);
    try {
      const payload: {
        taskId: number;
        body: string;
        anchor?: { quote: string; prefix?: string; suffix?: string };
      } = {
        taskId,
        body: draft.trim(),
      };
      if (pendingQuote) {
        payload.anchor = { quote: pendingQuote };
      }
      const res = await fetch("/api/updates/comment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });
      const json = (await res.json()) as
        | { ok: true; id: string }
        | { ok: false; error: string };
      if (!json.ok) {
        setStatus({ kind: "err", text: json.error });
        return;
      }
      setStatus({ kind: "ok", text: `Posted as ${json.id}.` });
      setDraft("");
      setPendingQuote(null);
      await onPosted();
      // Claude's auto-reply lands a few seconds later via the
      // fire-and-forget on the server side. Poll a couple of times to
      // surface it without the user having to refresh.
      for (const delayMs of [3_000, 8_000, 20_000]) {
        setTimeout(() => {
          void onPosted();
        }, delayMs);
      }
    } catch (e) {
      setStatus({
        kind: "err",
        text: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setPosting(false);
    }
  }

  return (
    <div className="space-y-2 rounded border border-stone-200 bg-white p-3">
      {pendingQuote && (
        <div className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-2 py-1.5 text-xs">
          <div className="flex-1">
            <div className="font-medium text-amber-900">
              Commenting on selection:
            </div>
            <blockquote className="mt-0.5 line-clamp-3 italic text-amber-950">
              &ldquo;
              {pendingQuote.length > 200
                ? pendingQuote.slice(0, 200) + "…"
                : pendingQuote}
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
      )}
      <textarea
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          setStatus(null);
        }}
        disabled={posting}
        placeholder={
          pendingQuote
            ? "Comment on this selection (markdown)."
            : "Add a comment (markdown). Select text in the body above to anchor it."
        }
        rows={3}
        className="w-full resize-y rounded border border-stone-300 bg-white px-2 py-1.5 text-sm font-mono"
      />
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void onSubmit()}
          disabled={posting || !draft.trim()}
          className="rounded bg-stone-900 px-3 py-1.5 text-sm font-medium text-white disabled:bg-stone-300"
        >
          {posting ? "…" : pendingQuote ? "Post anchored comment" : "Post comment"}
        </button>
        {status && (
          <span
            className={
              status.kind === "ok"
                ? "text-xs text-emerald-700"
                : "text-xs text-red-700"
            }
          >
            {status.text}
          </span>
        )}
      </div>
    </div>
  );
}
