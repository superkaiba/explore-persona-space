"use client";

/**
 * One feed card for /log. Renders the header always; the body + comment
 * thread lazy-mount on expand. Reuses the existing `CommentList` (now
 * with per-row Reply wiring routed at `/api/log/comment`) and pairs it
 * with a top-level composer for posting new threads.
 *
 * Default collapse:
 *   - clean-result with `classification: "not-useful"` → collapsed
 *   - everything else → expanded
 *
 * Comment count is lazy-fetched on first expand to avoid an N-card
 * round-trip on initial load.
 */
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { ChevronDown, ChevronRight, MessageSquare } from "lucide-react";
import type { FeedItem } from "@/lib/logs";
import type { TaskComment } from "@/lib/tasks";
import {
  CommentList,
  type PendingPlaceholder,
  type PostReplyResult,
  type ReplyWiring,
} from "@/app/tasks/[id]/CommentList";

// Refresh-poll cadence after an `@claude` post. The placeholder flips to
// `state: "error"` one tick past the final poll so the dismiss-X surfaces
// and the user isn't stuck staring at a forever-spinner when the sidecar
// failed server-side (sidecar only console.warn's; never persists a row).
// Matches CardCommentBox's PENDING_REPLY_TIMEOUT_MS (60s) loosely — we
// use 50s here because our last poll is 45s and we want the error to land
// shortly after the last refresh confirms the reply didn't arrive.
const PENDING_REFRESH_DELAYS_MS = [3_000, 8_000, 20_000, 45_000] as const;
const PENDING_REPLY_TIMEOUT_MS = 50_000;

export function LogCard({
  item,
  currentUserEmail,
}: {
  item: FeedItem;
  currentUserEmail: string | null;
}) {
  const defaultCollapsed =
    item.kind === "clean-result" && item.classification === "not-useful";
  const [expanded, setExpanded] = useState(!defaultCollapsed);
  const [comments, setComments] = useState<TaskComment[]>([]);
  const [commentsLoaded, setCommentsLoaded] = useState(false);
  const [commentCount, setCommentCount] = useState<number | null>(null);
  // Per-row spinner placeholders for in-thread Reply auto-replies. Keyed
  // by the user comment id that triggered the reply so CommentList can
  // render the spinner directly under it (same convention as
  // CardCommentBox on /updates).
  const [pending, setPending] = useState<PendingPlaceholder[]>([]);
  // Timer handles per pending placeholder (4 refresh polls + 1 error-flip).
  // Tracked so we can (a) cancel everything on unmount so React doesn't
  // warn about setState-on-unmounted-component when the user collapses
  // the card mid-poll, and (b) cancel the error-flip if the real reply
  // row lands before 50s. Keyed by `replyId` so two back-to-back @claude
  // comments get independent timer tracks.
  const pendingTimersRef = useRef<Map<string, number[]>>(new Map());

  // For clean-results, the comment thread lives on the task row
  // (tasks/<N>/comments.jsonl) and is reachable via the /tasks/<id>
  // page. We don't surface comments inline on the /log card; clicking
  // through gives the mentor the full task surface (plan, events,
  // existing anchored comments). For log entries, comments live at
  // logs/comments/<entryId>.jsonl and we render them inline.
  const supportsInlineComments = item.kind !== "clean-result";

  const refresh = useCallback(async () => {
    if (!supportsInlineComments) {
      setCommentCount(0);
      setCommentsLoaded(true);
      return;
    }
    try {
      const res = await fetch(
        `/api/log/comment?entryId=${encodeURIComponent(item.entryId)}`,
        { cache: "no-store", credentials: "same-origin" },
      );
      if (res.status === 401) {
        setComments([]);
        setCommentCount(0);
        setCommentsLoaded(true);
        return;
      }
      const json = (await res.json()) as
        | { ok: true; comments: TaskComment[] }
        | { ok: false; error: string };
      if (!json.ok) {
        setComments([]);
        setCommentCount(0);
        setCommentsLoaded(true);
        return;
      }
      setComments(json.comments);
      setCommentCount(json.comments.length);
      setCommentsLoaded(true);
    } catch {
      setCommentCount(0);
      setCommentsLoaded(true);
    }
  }, [item.entryId, supportsInlineComments]);

  // Lazy-load comments the first time the card is expanded. The hook's
  // setState lives inside `refresh` (which awaits a fetch before
  // calling setComments) so it doesn't run synchronously in the effect
  // body — but the react-hooks/set-state-in-effect rule still flags it.
  // Matches the same pattern used by /components/updates/CardCommentBox.
  useEffect(() => {
    if (expanded && !commentsLoaded && supportsInlineComments) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      void refresh();
    }
  }, [expanded, commentsLoaded, supportsInlineComments, refresh]);

  const handleToggle = useCallback(() => setExpanded((v) => !v), []);

  // Wire up per-row Reply for /log entries. Routes the POST to
  // /api/log/comment with `entryId` (not `taskId`) so the server keys
  // the comment to the right log file. Skipped on clean-result cards
  // because those don't render an inline thread (`supportsInlineComments`
  // is false) — the rail/CommentList only mounts on log entries.
  const postReply = useCallback(
    async (parentId: string, body: string): Promise<PostReplyResult> => {
      try {
        const res = await fetch("/api/log/comment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({ entryId: item.entryId, body, in_reply_to: parentId }),
        });
        return (await res.json()) as PostReplyResult;
      } catch (e) {
        return { ok: false, error: e instanceof Error ? e.message : String(e) };
      }
    },
    [item.entryId],
  );

  const replyWiring = useMemo<ReplyWiring | undefined>(() => {
    if (!currentUserEmail || !supportsInlineComments) return undefined;
    return {
      currentUserEmail,
      onPosted: refresh,
      onPendingStart: (parentId, replyId) => {
        setPending((cur) => {
          // Guard against duplicate starts for the same replyId so
          // back-to-back triggers don't stack spinners under the same row.
          if (cur.some((p) => p.id === replyId)) return cur;
          return [
            ...cur,
            { id: replyId, parentId, startedAt: Date.now(), state: "pending" },
          ];
        });
        // Schedule 4 refresh polls AND a final error-flip. Track every
        // timer id so unmount / dismiss / reply-landed can cancel cleanly.
        const ids: number[] = [];
        for (const delayMs of PENDING_REFRESH_DELAYS_MS) {
          ids.push(
            window.setTimeout(() => {
              void refresh();
            }, delayMs),
          );
        }
        // Final terminal state: if the reply hasn't landed in `comments`
        // by PENDING_REPLY_TIMEOUT_MS, flip the placeholder to "error" so
        // PendingItem renders the dismiss-X (the spinner-state hides it).
        // The reactive effect below clears placeholders whose reply DID
        // land — so this flip only fires when the sidecar genuinely
        // didn't persist a reply row.
        ids.push(
          window.setTimeout(() => {
            setPending((cur) =>
              cur.map((p) =>
                p.id === replyId && p.state === "pending"
                  ? { ...p, state: "error" as const }
                  : p,
              ),
            );
          }, PENDING_REPLY_TIMEOUT_MS),
        );
        pendingTimersRef.current.set(replyId, ids);
      },
      pending,
      onPendingDismiss: (replyId) => {
        const ids = pendingTimersRef.current.get(replyId);
        if (ids) {
          for (const t of ids) window.clearTimeout(t);
          pendingTimersRef.current.delete(replyId);
        }
        setPending((cur) => cur.filter((p) => p.id !== replyId));
      },
      postReply,
    };
  }, [currentUserEmail, supportsInlineComments, refresh, pending, postReply]);

  // Drop pending placeholders once the corresponding reply row has
  // landed in `comments` (matches CardCommentBox's dismissal logic).
  // Also cancel that placeholder's timer track so the deferred
  // error-flip doesn't fire on a row that already succeeded.
  useEffect(() => {
    if (pending.length === 0) return;
    const presentIds = new Set(comments.map((c) => c.id));
    const next = pending.filter((p) => !presentIds.has(p.id));
    if (next.length !== pending.length) {
      for (const p of pending) {
        if (presentIds.has(p.id)) {
          const ids = pendingTimersRef.current.get(p.id);
          if (ids) {
            for (const t of ids) window.clearTimeout(t);
            pendingTimersRef.current.delete(p.id);
          }
        }
      }
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPending(next);
    }
  }, [comments, pending]);

  // Cancel every outstanding pending-placeholder timer on unmount. Without
  // this, collapsing the card (or navigating away) before the 50s timeout
  // would let the setTimeout callbacks fire on an unmounted component,
  // producing React's "setState on unmounted component" warning.
  useEffect(() => {
    const timers = pendingTimersRef.current;
    return () => {
      for (const ids of timers.values()) {
        for (const t of ids) window.clearTimeout(t);
      }
      timers.clear();
    };
  }, []);

  return (
    <article
      className={`overflow-hidden rounded-lg border bg-white ${
        defaultCollapsed ? "border-stone-200/70" : "border-stone-200"
      }`}
    >
      <CardHeader
        item={item}
        commentCount={commentCount}
        expanded={expanded}
        onToggle={handleToggle}
        supportsInlineComments={supportsInlineComments}
      />
      {expanded && (
        <div className="border-t border-stone-100 p-4 sm:p-5">
          <CardBody item={item} />
          {item.kind === "clean-result" && (
            <p className="mt-4 text-xs text-stone-500">
              Full task surface (plan, events, anchored comments) at{" "}
              <Link
                href={`/tasks/${item.taskId}`}
                className="font-medium text-stone-700 underline hover:text-stone-900"
              >
                /tasks/{item.taskId}
              </Link>
              .
            </p>
          )}
          {supportsInlineComments && (
            <div className="mt-5 space-y-3 border-t border-stone-100 pt-4">
              <h3 className="text-sm font-semibold tracking-tight text-stone-800">
                Comments {commentCount !== null && `· ${commentCount}`}
              </h3>
              {currentUserEmail ? (
                <Composer
                  entryId={item.entryId}
                  currentUserEmail={currentUserEmail}
                  onPosted={refresh}
                />
              ) : (
                <p className="rounded border border-dashed border-stone-300 bg-white px-3 py-2 text-xs text-stone-500">
                  Sign in to comment on this entry.
                </p>
              )}
              {commentsLoaded ? (
                <CommentList
                  comments={comments}
                  inline
                  onDelete={
                    currentUserEmail
                      ? (commentId) => void deleteComment(item.entryId, commentId, refresh)
                      : undefined
                  }
                  reply={replyWiring}
                />
              ) : (
                <p className="text-xs text-stone-400">Loading comments…</p>
              )}
            </div>
          )}
        </div>
      )}
    </article>
  );
}

async function deleteComment(
  entryId: string,
  commentId: string,
  onDone: () => Promise<void>,
) {
  try {
    const res = await fetch("/api/log/comment", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ entryId, commentId }),
    });
    if (!res.ok) {
      const json = (await res.json().catch(() => ({}))) as { error?: string };
      window.alert(`Delete failed: ${json.error ?? res.statusText}`);
      return;
    }
    await onDone();
  } catch (e) {
    window.alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

/* -------------------------------------------------------------------------- *
 * Header / body / composer atoms.
 * -------------------------------------------------------------------------- */

function CardHeader({
  item,
  commentCount,
  expanded,
  onToggle,
  supportsInlineComments,
}: {
  item: FeedItem;
  commentCount: number | null;
  expanded: boolean;
  onToggle: () => void;
  supportsInlineComments: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-stone-50 sm:px-5"
      aria-expanded={expanded}
    >
      <span className="text-stone-400">
        {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </span>
      <KindBadge kind={item.kind} />
      <time className="font-mono text-xs tabular-nums text-stone-500">{item.date}</time>
      <span className="flex-1 truncate text-sm font-medium leading-snug text-stone-900">
        {item.title}
      </span>
      <span className="flex items-center gap-2">
        {item.kind === "clean-result" && (
          <ClassificationBadge classification={item.classification} />
        )}
        {supportsInlineComments && (
          <span
            className="inline-flex items-center gap-1 rounded bg-stone-100 px-1.5 py-0.5 text-[11px] text-stone-600"
            title="Comments"
          >
            <MessageSquare className="h-3 w-3" />
            {commentCount ?? "—"}
          </span>
        )}
      </span>
    </button>
  );
}

function CardBody({ item }: { item: FeedItem }) {
  return (
    <div className="prose prose-stone max-w-none sm:prose-base">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeHighlight]}
        components={{
          // Clean-result bodies repeat the title as an H1 (per the
          // verify_task_body.py spec); suppress it since the card
          // header already shows the title.
          h1: () => null,
          // `## Figure` is a structural label in the clean-result spec
          // but adds no signal to the rendered view.
          h2: ({ children, ...rest }) => {
            const text = Array.isArray(children)
              ? children.join("")
              : String(children ?? "");
            if (text.trim() === "Figure") return null;
            return <h2 {...rest}>{children}</h2>;
          },
        }}
      >
        {item.body}
      </ReactMarkdown>
    </div>
  );
}

function Composer({
  entryId,
  currentUserEmail,
  onPosted,
}: {
  entryId: string;
  currentUserEmail: string;
  onPosted: () => Promise<void>;
}) {
  void currentUserEmail; // surface the sign-in gate at the caller, not here
  const [draft, setDraft] = useState("");
  const [posting, setPosting] = useState(false);
  const [status, setStatus] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  // Hold the 4 poll timers scheduled after submit() so we can cancel them
  // on unmount (e.g. when the user collapses the card before the 45s
  // poll fires). Without this, the deferred `void onPosted()` calls
  // would invoke setState on an unmounted component.
  const pollTimersRef = useRef<number[]>([]);

  useEffect(() => {
    const timers = pollTimersRef.current;
    return () => {
      for (const t of timers) window.clearTimeout(t);
      timers.length = 0;
    };
  }, []);

  async function submit() {
    if (!draft.trim() || posting) return;
    setPosting(true);
    setStatus(null);
    try {
      const res = await fetch("/api/log/comment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ entryId, body: draft.trim() }),
      });
      const json = (await res.json()) as
        | { ok: true; id: string; will_reply?: boolean }
        | { ok: false; error: string };
      if (!json.ok) {
        setStatus({ kind: "err", text: json.error });
        return;
      }
      setStatus({
        kind: "ok",
        text: json.will_reply ? "Posted. Claude is replying…" : "Posted.",
      });
      setDraft("");
      await onPosted();
      // Auto-reply lands a few seconds later via fire-and-forget on
      // the server side. Poll a couple of times so the reply surfaces
      // without a manual refresh — same cadence as CardCommentBox.
      // Tracked in `pollTimersRef` so unmount cancels them cleanly.
      for (const delayMs of PENDING_REFRESH_DELAYS_MS) {
        pollTimersRef.current.push(
          window.setTimeout(() => {
            void onPosted();
          }, delayMs),
        );
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
      <textarea
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          setStatus(null);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            void submit();
          }
        }}
        disabled={posting}
        placeholder="Add a comment (Enter to post, Shift+Enter for newline). @claude to summon a reply."
        rows={3}
        className="w-full resize-y rounded border border-stone-300 bg-white px-2 py-1.5 text-sm font-mono"
      />
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void submit()}
          disabled={posting || !draft.trim()}
          className="rounded bg-stone-900 px-3 py-1.5 text-sm font-medium text-white disabled:bg-stone-300"
        >
          {posting ? "…" : "Post comment"}
        </button>
        {status && (
          <span
            className={
              status.kind === "ok" ? "text-xs text-emerald-700" : "text-xs text-red-700"
            }
          >
            {status.text}
          </span>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * Badges
 * -------------------------------------------------------------------------- */

function KindBadge({ kind }: { kind: FeedItem["kind"] }) {
  const palette: Record<FeedItem["kind"], string> = {
    daily: "bg-sky-50 text-sky-700",
    weekly: "bg-indigo-50 text-indigo-700",
    ideation: "bg-violet-50 text-violet-700",
    "clean-result": "bg-emerald-50 text-emerald-700",
  };
  const label: Record<FeedItem["kind"], string> = {
    daily: "Daily",
    weekly: "Weekly",
    ideation: "Ideation",
    "clean-result": "Result",
  };
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${palette[kind]}`}>
      {label[kind]}
    </span>
  );
}

function ClassificationBadge({
  classification,
}: {
  classification: "useful" | "not-useful" | "pending";
}) {
  const cls =
    classification === "useful"
      ? "bg-emerald-50 text-emerald-700"
      : classification === "not-useful"
        ? "bg-rose-50 text-rose-700"
        : "bg-stone-100 text-stone-700";
  const label =
    classification === "useful"
      ? "useful"
      : classification === "not-useful"
        ? "not useful"
        : "pending";
  return (
    <span className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${cls}`}>
      {label}
    </span>
  );
}
