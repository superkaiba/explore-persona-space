"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  AnchoredCommentsProvider,
  useAnchoredComments,
} from "@/app/tasks/[id]/AnchoredCommentsContext";
import { CommentableBody } from "@/app/tasks/[id]/CommentableBody";
import { CommentList } from "@/app/tasks/[id]/CommentList";
import { TocSidebar } from "@/components/updates/TocSidebar";
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

/**
 * Client-side placeholder we inject when `@claude` is detected so the
 * mentor sees "Claude is thinking…" immediately. The placeholder
 * shares the `id` with the eventual server-persisted reply, so when the
 * real row arrives via polling it naturally replaces the placeholder.
 * On timeout, the placeholder flips to an error variant that can be
 * dismissed via DELETE (replies are deletable by any signed-in user).
 */
type PendingPlaceholder = {
  id: string;
  parentId: string;
  startedAt: number;
  state: "pending" | "error";
};

const PENDING_REPLY_TIMEOUT_MS = 60_000;

export function CardCommentBox({
  taskId,
  body,
  currentUserEmail,
  layout = "inline",
  onUnaddressedChange,
  refreshNonce = 0,
}: {
  taskId: number;
  body: string;
  currentUserEmail: string | null;
  layout?: Layout;
  /**
   * Called whenever the unaddressed `anchor-comment` count changes. Used
   * by the modal full-view to enable/disable the "Address comments"
   * button without an extra round-trip.
   */
  onUnaddressedChange?: (count: number) => void;
  /**
   * Bumping this from the parent forces a comments re-fetch. Used after
   * `/api/updates/address-comments` rewrites comments.jsonl so the
   * addressed-badges + synthesis reply land without a hard reload.
   */
  refreshNonce?: number;
}) {
  const [comments, setComments] = useState<FetchedComment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState<PendingPlaceholder[]>([]);
  // Timer handles per pending placeholder so we can flip them to "error"
  // if the server takes >60s.
  const pendingTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

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
  }, [refresh, refreshNonce]);

  // Clear any pending placeholder whose real row has landed (matched by
  // id). Also report the unaddressed-anchor-comment count up to the
  // parent so the "Address comments" button can enable/disable.
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
    if (onUnaddressedChange) {
      const unaddressed = comments.filter(
        (c) =>
          c.kind === "anchor-comment" &&
          (c as Record<string, unknown>).addressed !== true,
      ).length;
      onUnaddressedChange(unaddressed);
    }
  }, [comments, onUnaddressedChange]);

  // Cleanup timers on unmount.
  useEffect(() => {
    const timers = pendingTimersRef.current;
    return () => {
      for (const t of timers.values()) clearTimeout(t);
      timers.clear();
    };
  }, []);

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

  const onPendingStart = useCallback((parentId: string, replyId: string) => {
    setPending((prev) => {
      if (prev.some((p) => p.id === replyId)) return prev;
      return [...prev, { id: replyId, parentId, startedAt: Date.now(), state: "pending" }];
    });
    const timer = setTimeout(() => {
      setPending((prev) =>
        prev.map((p) => (p.id === replyId ? { ...p, state: "error" as const } : p)),
      );
    }, PENDING_REPLY_TIMEOUT_MS);
    pendingTimersRef.current.set(replyId, timer);
  }, []);

  const onPendingDismiss = useCallback(
    async (replyId: string) => {
      const timer = pendingTimersRef.current.get(replyId);
      if (timer) {
        clearTimeout(timer);
        pendingTimersRef.current.delete(replyId);
      }
      setPending((prev) => prev.filter((p) => p.id !== replyId));
      // Best-effort DELETE in case the row eventually lands.
      try {
        await fetch("/api/updates/comment", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({ taskId, commentId: replyId }),
        });
      } catch {
        // best-effort
      }
    },
    [taskId],
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
        pending={pending}
        onPendingStart={onPendingStart}
        onPendingDismiss={onPendingDismiss}
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
  pending,
  onPendingStart,
  onPendingDismiss,
}: {
  taskId: number;
  body: string;
  comments: TaskComment[];
  currentUserEmail: string | null;
  layout: Layout;
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<void>;
  pending: PendingPlaceholder[];
  onPendingStart: (parentId: string, replyId: string) => void;
  onPendingDismiss: (replyId: string) => Promise<void>;
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

  // Pending placeholders are rendered as a sibling block (one per
  // outstanding `@claude` reply). CommentList itself stays unchanged —
  // it just renders persisted rows from comments.jsonl.
  const pendingBlock = pending.length > 0 ? (
    <ul className="space-y-2">
      {pending.map((p) => (
        <PendingReplyCard
          key={p.id}
          placeholder={p}
          onDismiss={() => void onPendingDismiss(p.id)}
        />
      ))}
    </ul>
  ) : null;

  if (layout === "rail") {
    return (
      <div className="grid gap-6 lg:grid-cols-[200px_minmax(0,1fr)_360px]">
        {/* TocSidebar hides itself (`hidden lg:block`) at narrow widths so
            the 2-col layout still fits on smaller modals. */}
        <TocSidebar body={body} taskId={taskId} />
        <div className="min-w-0">
          <CommentableBody
            body={body}
            isLegacyHtml={false}
            enableCollapsibleSections
            taskId={taskId}
          />
        </div>
        <aside className="min-w-0 relative">
          <CardComposer
            taskId={taskId}
            currentUserEmail={currentUserEmail}
            onPosted={onRefresh}
            onPendingStart={onPendingStart}
          />
          <div className="mt-4 space-y-3">
            {pendingBlock}
            {loading ? null : error ? (
              <CommentsError error={error} />
            ) : (
              /* Drop `inline` so CommentList's useLayoutEffect alignment
                 math runs — pushes each anchored comment down so its top
                 matches the anchor mark's Y position in the body. */
              <CommentList comments={comments} onDelete={onDelete} />
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
        onPendingStart={onPendingStart}
      />
      {pendingBlock}
      {loading ? null : error ? (
        <CommentsError error={error} />
      ) : (
        <CommentList comments={comments} inline onDelete={onDelete} />
      )}
    </div>
  );
}

function PendingReplyCard({
  placeholder,
  onDismiss,
}: {
  placeholder: PendingPlaceholder;
  onDismiss: () => void;
}) {
  if (placeholder.state === "error") {
    return (
      <li className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
        <div className="flex items-start justify-between gap-2">
          <span>
            Claude didn&rsquo;t respond — try again or check the server logs.
          </span>
          <button
            type="button"
            onClick={onDismiss}
            className="rounded p-0.5 text-amber-700 hover:bg-amber-100 hover:text-amber-900"
            aria-label="Dismiss"
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      </li>
    );
  }
  return (
    <li className="rounded border border-stone-200 bg-stone-50 px-3 py-2 text-xs animate-pulse">
      <div className="flex items-center gap-2 text-stone-600">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <span>Claude is thinking…</span>
      </div>
    </li>
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
  onPendingStart,
}: {
  taskId: number;
  currentUserEmail: string | null;
  onPosted: () => Promise<void>;
  onPendingStart: (parentId: string, replyId: string) => void;
}) {
  const { pendingQuote, setPendingQuote } = useAnchoredComments();
  const [draft, setDraft] = useState("");
  const [posting, setPosting] = useState(false);
  const [status, setStatus] = useState<
    { kind: "ok" | "err"; text: string } | null
  >(null);
  // While the user is composing an anchored comment, push the composer
  // down so it sits next to the pending highlight in the body — instead
  // of staying pinned at the top of the rail. Recomputes whenever the
  // pendingQuote changes; resets to 0 once posted/cleared.
  const composerRef = useRef<HTMLDivElement>(null);
  const [pendingMarginTop, setPendingMarginTop] = useState(0);
  useEffect(() => {
    if (!pendingQuote) {
      setPendingMarginTop(0);
      return;
    }
    const compute = () => {
      const el = composerRef.current;
      if (!el) return;
      const aside = el.closest("aside");
      const mark = document.querySelector<HTMLElement>(
        "mark[data-anchor-pending]",
      );
      if (!aside || !mark) return;
      const asideTop = aside.getBoundingClientRect().top;
      const markTop = mark.getBoundingClientRect().top;
      setPendingMarginTop(Math.max(0, markTop - asideTop));
    };
    // Defer one frame so the mark has finished wrapping.
    const t = window.setTimeout(compute, 0);
    return () => window.clearTimeout(t);
  }, [pendingQuote]);

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
        | {
            ok: true;
            id: string;
            ts?: string;
            will_reply?: boolean;
            pending_reply_id?: string;
          }
        | { ok: false; error: string };
      if (!json.ok) {
        setStatus({ kind: "err", text: json.error });
        return;
      }
      setStatus({ kind: "ok", text: `Posted as ${json.id}.` });
      setDraft("");
      setPendingQuote(null);
      if (json.will_reply && json.pending_reply_id) {
        onPendingStart(json.id, json.pending_reply_id);
      }
      await onPosted();
      // Claude's auto-reply lands a few seconds later via the
      // fire-and-forget on the server side. Poll a couple of times to
      // surface it without the user having to refresh. Body-edit + code-
      // edit paths can take 30-60s, so extend the tail of the poll.
      for (const delayMs of [3_000, 8_000, 20_000, 45_000]) {
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
    <div
      ref={composerRef}
      style={pendingMarginTop > 0 ? { marginTop: `${pendingMarginTop}px` } : undefined}
      className="space-y-2 rounded border border-stone-200 bg-white p-3"
    >
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
        onKeyDown={(e) => {
          // Enter posts; Shift+Enter (or Cmd/Ctrl+Enter) inserts a newline.
          if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            void onSubmit();
          }
        }}
        disabled={posting}
        placeholder={
          pendingQuote
            ? "Comment on this selection (Enter to post, Shift+Enter for newline). @claude to summon a reply."
            : "Add a comment (Enter to post, Shift+Enter for newline). @claude to summon a reply. Select text above to anchor."
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
