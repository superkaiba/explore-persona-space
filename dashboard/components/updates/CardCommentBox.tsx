"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
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
  // Archived comments are excluded — once a comment is auto-archived on
  // address, it no longer contributes to the "needs addressing" count.
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
          (c as Record<string, unknown>).addressed !== true &&
          (c as Record<string, unknown>).archived !== true,
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

  // Split comments into visible vs archived. Archived anchor-comments
  // (auto-set by /api/updates/address-comments alongside `addressed: true`)
  // are hidden from the main rail and surfaced in a collapsible
  // "Archived (N)" section at the bottom. Replies inherit their parent's
  // archived state — if the parent anchor-comment is archived, its
  // synthesis reply AND any user follow-ups (whose `in_reply_to` points
  // at the synthesis reply, NOT the root) move with it.
  //
  // The closure is computed iteratively rather than one-hop because a
  // thread can be 3+ deep: archived root -> claude reply -> user
  // follow-up. A one-hop filter would leak the user follow-up into the
  // visible rail where `CommentList` silently drops it (its parent was
  // filtered out), while it still inflated the <mark> anchors and the
  // unaddressed count. 64-hop cap is a sanity bound — real threads are
  // O(10).
  const archivedIds = useMemo(() => {
    const closed = new Set<string>();
    for (const c of comments) {
      if (
        c.kind === "anchor-comment" &&
        (c as Record<string, unknown>).archived === true
      ) {
        closed.add(c.id);
      }
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

  const archivedComments = useMemo(
    () => comments.filter((c) => archivedIds.has(c.id)),
    [comments, archivedIds],
  );

  const anchors = useMemo(
    () =>
      visibleComments
        // Reply rows share their parent's anchor; don't double-wrap. Only
        // user-authored anchor-comment rows contribute a <mark>. Archived
        // comments are filtered out via `visibleComments` so their
        // highlights disappear from the body too.
        .filter(
          (c) =>
            c.kind === "anchor-comment" &&
            typeof c.anchor?.quote === "string" &&
            c.anchor!.quote.length > 0,
        )
        .map((c) => ({ id: c.id, quote: c.anchor!.quote })),
    [visibleComments],
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
        comments={visibleComments as TaskComment[]}
        archivedComments={archivedComments as TaskComment[]}
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
  archivedComments,
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
  /** Comments visible in the main rail: not archived, and not a reply to
   *  an archived parent. Parent handles the split so this component can
   *  treat `comments` as "the rail's complete dataset". */
  comments: TaskComment[];
  /** Archived anchor-comments + their replies, surfaced in a collapsible
   *  "Archived (N)" strip beneath the rail. Empty array hides the strip. */
  archivedComments: TaskComment[];
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

  const handleUnarchive = useCallback(
    async (commentId: string) => {
      try {
        const res = await fetch("/api/updates/unarchive-comment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({ taskId, commentId }),
        });
        if (!res.ok) {
          const json = (await res.json().catch(() => ({}))) as { error?: string };
          window.alert(`Unarchive failed: ${json.error ?? res.statusText}`);
          return;
        }
        await onRefresh();
      } catch (e) {
        window.alert(
          `Unarchive failed: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
    [taskId, onRefresh],
  );

  // The Unarchive BUTTON is shown only to signed-in editors — the
  // unarchive route is editor-gated, so anonymous viewers would see a
  // button that 401s. Anonymous viewers still see the archived strip
  // itself (so they know thread context exists), just without the
  // button. `ArchivedStrip` reads the absence of `onUnarchive` as
  // "render in read-only mode".
  const onUnarchive = currentUserEmail ? handleUnarchive : undefined;

  // The CommentList delete control is wired per-row, but the server
  // also enforces author-match. Show the button on every row when the
  // viewer is signed in; the server returns 403 if they aren't the
  // author of that particular row.
  const onDelete = currentUserEmail ? handleDelete : undefined;

  // While the user is composing an anchored comment, push the composer
  // down so it sits next to the highlighted span. State lives here at
  // the parent so we can pass it to CommentList as an alignment dep —
  // CommentList re-runs its anchor alignment when the ul's docTop
  // changes (otherwise the comments would visibly shift when composer
  // moves).
  const { pendingQuote: railPendingQuote } = useAnchoredComments();
  const [composerMarginTop, setComposerMarginTop] = useState(0);
  useEffect(() => {
    if (!railPendingQuote) {
      setComposerMarginTop(0);
      return;
    }
    const compute = () => {
      const aside = document.querySelector<HTMLElement>("aside[data-rail-aside]");
      const mark = document.querySelector<HTMLElement>("mark[data-anchor-pending]");
      if (!aside || !mark) return;
      const offset = mark.getBoundingClientRect().top - aside.getBoundingClientRect().top;
      setComposerMarginTop(Math.max(0, offset));
    };
    const t = window.setTimeout(compute, 0);
    return () => window.clearTimeout(t);
  }, [railPendingQuote]);

  // Pending placeholders flow INTO CommentList so each spinner renders
  // beneath the user comment that triggered it (instead of one global
  // stack at the top of the rail). Reply wiring lets each row host its
  // own inline Reply composer + per-thread spinner.
  const replyWiring = currentUserEmail
    ? {
        taskId,
        currentUserEmail,
        onPosted: onRefresh,
        onPendingStart,
        pending,
        onPendingDismiss: (id: string) => void onPendingDismiss(id),
      }
    : undefined;

  if (layout === "rail") {
    return (
      /* Container query: the grid responds to the MODAL's width, not the
         window width. Was `lg:grid-cols-[...]` (window-width 1024px) so
         in fullscreen on a narrow browser the rail stacked below the body
         off-screen. `@container` + `@4xl:` activates the 3-col grid as
         soon as the modal panel is ≥ 56rem (~896px), which covers
         fullscreen on any reasonable screen. Below that the rail stacks
         under the body — still accessible, just no side-by-side. */
      <div className="@container">
        <div className="grid gap-6 @4xl:grid-cols-[200px_minmax(0,1fr)_360px]">
        {/* TocSidebar hides itself at narrow container widths via its
            internal `hidden @4xl:block`. */}
        <TocSidebar body={body} taskId={taskId} />
        <div className="min-w-0">
          <CommentableBody
            body={body}
            isLegacyHtml={false}
            enableCollapsibleSections
            taskId={taskId}
          />
        </div>
        <aside data-rail-aside className="min-w-0 relative">
          <div style={composerMarginTop > 0 ? { marginTop: `${composerMarginTop}px` } : undefined}>
            <CardComposer
              taskId={taskId}
              currentUserEmail={currentUserEmail}
              onPosted={onRefresh}
              onPendingStart={onPendingStart}
            />
          </div>
          <div className="mt-4 space-y-3">
            {loading ? null : error ? (
              <CommentsError error={error} />
            ) : (
              /* Drop `inline` so CommentList's useLayoutEffect alignment
                 math runs — pushes each anchored comment down so its top
                 matches the anchor mark's Y position in the body. Pass
                 composerMarginTop as a re-alignment trigger so comments
                 stay viewport-stable when the composer pushes the ul.
                 `reply` wiring enables per-row Reply button + inline
                 composer + inline pending placeholders. */
              <CommentList
                comments={comments}
                onDelete={onDelete}
                alignmentNonce={composerMarginTop}
                reply={replyWiring}
              />
            )}
          </div>
          {!loading && !error && (
            <ArchivedStrip
              comments={archivedComments}
              onUnarchive={onUnarchive}
            />
          )}
        </aside>
        </div>
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
      {loading ? null : error ? (
        <CommentsError error={error} />
      ) : (
        <CommentList
          comments={comments}
          inline
          onDelete={onDelete}
          reply={replyWiring}
        />
      )}
      {!loading && !error && (
        <ArchivedStrip
          comments={archivedComments}
          onUnarchive={onUnarchive}
        />
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
 * Collapsible "Archived (N)" strip beneath the rail. Renders each archived
 * anchor-comment with its synthesis reply (and any user follow-ups) so the
 * thread context is preserved. Editors get an "Unarchive" button per
 * anchor-comment that POSTs to /api/updates/unarchive-comment, then the
 * parent's `refresh()` pulls the row back into the visible rail.
 *
 * Uses a native <details> element rather than the project's
 * CollapsiblePanel — that component is keyed to (taskId, itemKey) for
 * persistence and ties into the TOC expand-event system, neither of which
 * is useful here. <details> is one line, accessible, and collapsed by
 * default which is the right default for an "archive" affordance.
 *
 * The N badge in the header counts ANCHOR-COMMENTS only — replies that
 * moved with their parent are visible inside but don't inflate the count,
 * which would otherwise mislead the user about how many threads got
 * archived.
 */
function ArchivedStrip({
  comments,
  onUnarchive,
}: {
  comments: TaskComment[];
  onUnarchive?: (commentId: string) => Promise<void> | void;
}) {
  // Group archived rows: a thread root is an anchor-comment WITHOUT
  // `in_reply_to` (the original mentor comment that got archived).
  // Anchor-comments that ARE replies (user follow-ups whose parent is
  // an archived reply) nest under their parent instead of appearing as
  // flat duplicate roots — this also keeps the `Archived (N)` count
  // honest (it counts threads, not rows). Anchor-comment roots sort by
  // ts oldest-first (matches archive ordering: first archived appears
  // at top); replies sort by ts within each thread.
  const { roots, repliesByParent } = useMemo(() => {
    const r: TaskComment[] = [];
    const rep: Record<string, TaskComment[]> = {};
    for (const c of comments) {
      if (c.kind === "anchor-comment" && !c.in_reply_to) {
        r.push(c);
      } else if (c.in_reply_to) {
        (rep[c.in_reply_to] ||= []).push(c);
      }
    }
    r.sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
    for (const k of Object.keys(rep)) {
      rep[k].sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
    }
    return { roots: r, repliesByParent: rep };
  }, [comments]);

  if (roots.length === 0) return null;

  return (
    <details className="mt-4 rounded border border-stone-200 bg-stone-50">
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-stone-600 hover:bg-stone-100">
        Archived ({roots.length})
      </summary>
      <ul className="space-y-2 border-t border-stone-200 px-3 py-3">
        {roots.map((root) => (
          <li
            key={root.id}
            className="rounded border border-stone-200 bg-white px-2.5 py-2 text-sm"
          >
            <ArchivedRow comment={root} onUnarchive={onUnarchive} />
            {(repliesByParent[root.id] ?? []).length > 0 && (
              <ArchivedReplies
                parentId={root.id}
                repliesByParent={repliesByParent}
              />
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

function ArchivedRow({
  comment,
  onUnarchive,
}: {
  comment: TaskComment;
  onUnarchive?: (commentId: string) => Promise<void> | void;
}) {
  const anchor = readArchivedAnchorQuote(comment);
  const [pending, setPending] = useState(false);
  return (
    <>
      <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-stone-500">
        <span className="font-mono text-stone-400">{comment.id}</span>
        <span className="font-medium text-stone-700">{comment.author}</span>
        <span className="rounded bg-stone-100 px-1 text-[10px] uppercase tracking-wide text-stone-600">
          {comment.kind}
        </span>
        <time className="ml-auto tabular-nums">{compactArchivedTs(comment.ts)}</time>
        {onUnarchive && (
          <button
            type="button"
            onClick={async () => {
              if (pending) return;
              setPending(true);
              try {
                await onUnarchive(comment.id);
              } finally {
                setPending(false);
              }
            }}
            disabled={pending}
            className="rounded border border-stone-300 bg-white px-1.5 py-0.5 text-[10px] font-medium text-stone-700 hover:bg-stone-100 disabled:opacity-50"
          >
            {pending ? "…" : "Unarchive"}
          </button>
        )}
      </div>
      {anchor && (
        <blockquote className="mb-1 border-l-2 border-amber-300 pl-2 text-[11px] italic text-stone-600">
          &ldquo;{anchor.length > 140 ? anchor.slice(0, 140) + "…" : anchor}&rdquo;
        </blockquote>
      )}
      <div className="prose prose-sm prose-stone max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw, rehypeHighlight]}
        >
          {comment.body}
        </ReactMarkdown>
      </div>
    </>
  );
}

function ArchivedReplies({
  parentId,
  repliesByParent,
}: {
  parentId: string;
  repliesByParent: Record<string, TaskComment[]>;
}) {
  const replies = repliesByParent[parentId];
  if (!replies || replies.length === 0) return null;
  return (
    <ul className="mt-2 space-y-1.5 border-l-2 border-stone-200 pl-3">
      {replies.map((c) => {
        const isClaudeReply =
          c.author === "claude" || c.kind === "anchor-comment-reply";
        return (
          <li
            key={c.id}
            className={`rounded border px-2.5 py-2 text-sm ${
              isClaudeReply
                ? "border-stone-200 bg-stone-50"
                : "border-stone-200 bg-white"
            }`}
          >
            <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-stone-500">
              <span className="font-medium text-stone-700">
                {isClaudeReply ? "Claude" : c.author}
              </span>
              <time className="ml-auto tabular-nums">{compactArchivedTs(c.ts)}</time>
            </div>
            <div className="prose prose-sm prose-stone max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw, rehypeHighlight]}
              >
                {c.body}
              </ReactMarkdown>
            </div>
            {/* Recurse: handles user follow-ups + Claude replies-of-replies. */}
            <ArchivedReplies
              parentId={c.id}
              repliesByParent={repliesByParent}
            />
          </li>
        );
      })}
    </ul>
  );
}

function readArchivedAnchorQuote(c: TaskComment): string | null {
  const a = (c as Record<string, unknown>).anchor;
  if (a && typeof a === "object" && a !== null) {
    const q = (a as { quote?: unknown }).quote;
    if (typeof q === "string" && q.trim()) return q;
  }
  return null;
}

function compactArchivedTs(ts: string): string {
  const m = ts.match(/^\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]}` : ts;
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
