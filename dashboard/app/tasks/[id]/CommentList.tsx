"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { Check, Loader2, X } from "lucide-react";
import type { TaskComment } from "@/lib/tasks";
import { useAnchoredComments } from "./AnchoredCommentsContext";

/**
 * Per-comment pending placeholder. Forwarded by the parent
 * (CardCommentBoxInner) so we can render "Claude is working…" cards
 * inline beneath the user comment that triggered them, instead of in
 * one global stack at the top of the rail.
 */
export type PendingPlaceholder = {
  id: string;
  parentId: string;
  startedAt: number;
  state: "pending" | "error";
};

/**
 * Result shape the ReplyComposer expects back from its POST. Mirrors the
 * `/api/updates/comment` and `/api/log/comment` response shapes — both
 * routes return `{ ok: true, id, will_reply?, pending_reply_id? }` on
 * success and `{ ok: false, error }` on failure.
 */
export type PostReplyResult =
  | { ok: true; id: string; ts?: string; will_reply?: boolean; pending_reply_id?: string }
  | { ok: false; error: string };

/**
 * Wiring needed to render the inline per-row Reply composer + Pending
 * placeholder cards. Optional — when omitted, CommentList renders in
 * read-only / delete-only mode (matches the original API).
 *
 * `postReply` is optional. When omitted (the original behavior), the
 * composer POSTs to `/api/updates/comment` with `{ taskId, body,
 * in_reply_to }`. When provided, the composer calls `postReply` instead,
 * letting non-task entities (e.g. /log entries) route the reply to their
 * own endpoint with their own entity key. `taskId` is also optional now;
 * callers that supply `postReply` typically don't have one.
 */
export type ReplyWiring = {
  taskId?: number;
  currentUserEmail: string;
  onPosted: () => Promise<void>;
  onPendingStart: (parentId: string, replyId: string) => void;
  pending?: PendingPlaceholder[];
  onPendingDismiss?: (replyId: string) => void;
  /**
   * Optional override for the reply POST. When provided, the inline
   * ReplyComposer calls this instead of fetching `/api/updates/comment`.
   * Returns the same response shape as the updates endpoint so the
   * pending-placeholder UX is identical regardless of which entity
   * the comment belongs to.
   */
  postReply?: (parentId: string, body: string) => Promise<PostReplyResult>;
};

/**
 * Sidebar comment list.
 *
 * Behaviors:
 *   - Comments are sorted so anchored ones appear in TEXT order (by the
 *     anchor's Y position in the body). Unanchored comments stack at the
 *     bottom in timestamp order.
 *   - Each anchored comment is *vertically aligned* with its anchor in
 *     the body via a computed `margin-top`. A useLayoutEffect runs after
 *     each render to align without flicker (sidebar/`rail` layout only —
 *     `inline` skips this so the comments flow naturally below the body).
 *   - Hovering a comment highlights two related sets:
 *       (a) the comment it replies to (`in_reply_to`)
 *       (b) any comments that reply TO this comment
 *     and also tells the body (via context) to darken the matching <mark>.
 *   - Clicking an anchored comment scrolls its <mark> into view.
 *   - When `onDelete` is provided, each comment renders a small ✕ that
 *     asks for native confirm() and then invokes the callback. The list
 *     does NOT decide who can delete — the caller is responsible for
 *     gating (e.g. only show the prop when the row.author matches the
 *     current user); the server enforces the actual permission.
 */
export function CommentList({
  comments,
  inline = false,
  onDelete,
  alignmentNonce = 0,
  reply,
}: {
  comments: TaskComment[];
  /**
   * `false` (default): sidebar layout used by /tasks/[id] — vertically
   * aligns each anchored comment with its anchor in the body via
   * margin-top adjustments.
   * `true`: inline-below-body layout used by /updates cards — stacks
   * comments in text-then-ts order with no alignment math.
   */
  inline?: boolean;
  /**
   * When provided, renders a ✕ button per comment that confirms via
   * native confirm() then calls `onDelete(commentId)`. The caller may
   * choose to only show this for rows the current user authored.
   */
  onDelete?: (commentId: string) => void;
  /**
   * Bump this number to force the anchor-alignment effect to re-run
   * even when `sorted` / `anchorTopById` haven't changed. Used by the
   * parent to re-align after the composer's marginTop pushes the ul
   * down — otherwise the comments visibly shift instead of staying
   * fixed at their anchor's viewport position.
   */
  alignmentNonce?: number;
  /**
   * When provided, each comment row renders a small "Reply" button that
   * opens an inline composer beneath it. Replies are posted with
   * `in_reply_to: <parent-id>` so they nest under the parent in the
   * same thread. Pending placeholders (`reply.pending`) render under
   * the comment whose `parentId` matches, instead of in one global
   * stack — keeps the spinner next to the thread it belongs to.
   */
  reply?: ReplyWiring;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  // `hoveredId` from context: set by CommentableBody when the user hovers
  // a <mark data-comment-id> in the body. We OR it with the local `hovered`
  // state so hovering the highlighted span lights up the matching rail
  // comment (the reverse direction was already wired the other way).
  const {
    hoveredId: hoveredFromBody,
    setHoveredId,
    requestScrollTo,
    anchorPositions,
  } = useAnchoredComments();
  const listRef = useRef<HTMLUListElement>(null);
  const effectiveHovered = hovered ?? hoveredFromBody;

  const anchorTopById = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of anchorPositions) m.set(p.id, p.top);
    return m;
  }, [anchorPositions]);

  // Build parent → ordered-replies map. Only direct children; replies of
  // replies cascade naturally when the recursion picks up their own
  // entry. Ordered by timestamp so threads read top-down.
  const repliesByParent = useMemo(() => {
    const idx: Record<string, TaskComment[]> = {};
    for (const c of comments) {
      if (c.in_reply_to) {
        (idx[c.in_reply_to] ||= []).push(c);
      }
    }
    for (const k of Object.keys(idx)) {
      idx[k].sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
    }
    return idx;
  }, [comments]);

  // Sort top-level comments: anchored first by anchor Y (= text order),
  // then unanchored by ts. Replies are rendered nested under their
  // parent — they NEVER appear in the top-level list.
  const sorted = useMemo(() => {
    const list = comments.filter((c) => !c.in_reply_to);
    list.sort((a, b) => {
      const aTop = anchorTopById.get(a.id);
      const bTop = anchorTopById.get(b.id);
      if (aTop !== undefined && bTop !== undefined) return aTop - bTop;
      if (aTop !== undefined) return -1; // anchored before unanchored
      if (bTop !== undefined) return 1;
      return (a.ts || "").localeCompare(b.ts || "");
    });
    return list;
  }, [comments, anchorTopById]);

  // Build child-index for hover highlight: { parent_cid -> [reply_cids] }.
  const childrenOf = useMemo(() => {
    const idx: Record<string, string[]> = {};
    for (const c of comments) {
      if (c.in_reply_to) {
        (idx[c.in_reply_to] ||= []).push(c.id);
      }
    }
    return idx;
  }, [comments]);

  const byId = useMemo(
    () => Object.fromEntries(comments.map((c) => [c.id, c])),
    [comments],
  );

  // Only ONE inline reply composer open at a time. Track its parent's
  // id; null = no composer. Cleared by the composer on successful post
  // or by the Cancel button. If the parent comment disappears from
  // `comments`, the composer naturally stops rendering — its `c.id`
  // is no longer in the .map() — so there's no separate cleanup
  // needed for that case.
  const [replyingTo, setReplyingTo] = useState<string | null>(null);

  // Group pending placeholders by parent id so each thread can render
  // its own spinner inline. The parent id stored on a placeholder is
  // the USER comment id that triggered the reply (set by
  // CardCommentBoxInner's onPendingStart) — so a top-level @claude
  // mention is parented to the top-level user row, and a follow-up
  // is parented to the user follow-up row.
  const pendingByParent = useMemo(() => {
    const idx: Record<string, PendingPlaceholder[]> = {};
    for (const p of reply?.pending ?? []) {
      (idx[p.parentId] ||= []).push(p);
    }
    return idx;
  }, [reply?.pending]);

  function isHighlighted(cid: string): boolean {
    const h = effectiveHovered;
    if (!h) return false;
    if (cid === h) return true;
    const hoveredC = byId[h];
    if (hoveredC?.in_reply_to === cid) return true;
    if ((childrenOf[h] || []).includes(cid)) return true;
    return false;
  }

  // Vertical alignment via ABSOLUTE positioning. Each anchored li gets
  // top = anchor's Y (relative to the list), so adding/removing a
  // comment doesn't reflow anything else and the layout doesn't slide.
  // Collision avoidance still runs (in JS, post-paint) — if two anchors
  // are close enough that comments would overlap, later ones get pushed
  // down so they don't sit on top of each other. Comments without an
  // anchor stack at the bottom in normal flow above the absolute layer.
  // Skipped entirely in inline mode.
  useLayoutEffect(() => {
    if (inline) return;
    const list = listRef.current;
    if (!list) return;
    const items = Array.from(list.querySelectorAll<HTMLLIElement>("li[data-cid]"));
    if (items.length === 0) return;
    // Clear any inline top set on a previous pass so unanchored items
    // return to normal flow when their anchor is removed.
    for (const li of items) {
      const cid = li.dataset.cid!;
      if (!anchorTopById.has(cid)) {
        li.style.position = "";
        li.style.top = "";
        li.style.left = "";
        li.style.right = "";
      }
    }
    const listDocTop = list.getBoundingClientRect().top + window.scrollY;
    let prevBottom = 0; // collision floor, in list-relative coords
    const GAP = 8;
    let maxBottom = 0;
    for (const li of items) {
      const cid = li.dataset.cid!;
      const anchorTop = anchorTopById.get(cid);
      if (anchorTop === undefined) continue;
      const height = li.getBoundingClientRect().height || li.offsetHeight;
      const relativeAnchor = Math.max(0, anchorTop - listDocTop);
      const desiredTop = Math.max(relativeAnchor, prevBottom + GAP);
      li.style.position = "absolute";
      li.style.top = `${desiredTop}px`;
      li.style.left = "0";
      li.style.right = "0";
      prevBottom = desiredTop + height;
      if (prevBottom > maxBottom) maxBottom = prevBottom;
    }
    // Reserve enough vertical space so the absolute children don't
    // collapse the ul's height — otherwise unanchored items below would
    // render on top of anchored ones.
    list.style.minHeight = `${maxBottom}px`;
  }, [sorted, anchorTopById, inline, alignmentNonce]);

  if (comments.length === 0) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-3 py-4 text-center text-xs text-stone-500">
        No comments yet.
      </p>
    );
  }

  return (
    <ul ref={listRef} className={`space-y-1.5 ${inline ? "" : "relative"}`}>
      {sorted.map((c) => {
        const highlighted = isHighlighted(c.id);
        const anchor = readAnchorQuote(c);
        const isAnchored = anchorTopById.has(c.id);
        const addressed = readAddressed(c);
        return (
          <li
            key={c.id}
            data-cid={c.id}
            onMouseEnter={() => {
              setHovered(c.id);
              setHoveredId(c.id);
            }}
            onMouseLeave={() => {
              setHovered(null);
              setHoveredId(null);
            }}
            onClick={() => {
              if (isAnchored) requestScrollTo(c.id);
            }}
            className={`rounded border px-2.5 py-2 text-sm transition-colors duration-100 ${
              isAnchored ? "cursor-pointer" : ""
            } ${
              highlighted
                ? "border-amber-300 bg-amber-50"
                : "border-stone-200 bg-white hover:border-stone-300"
            }`}
            title={isAnchored ? "Click to scroll the highlighted span into view" : undefined}
          >
            <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-stone-500">
              {addressed && (
                <span
                  className="inline-flex items-center gap-0.5 rounded bg-emerald-100 px-1 py-px text-[10px] font-medium uppercase tracking-wide text-emerald-800"
                  title={addressed.note || "Addressed by Claude"}
                >
                  <Check className="h-3 w-3" />
                  addressed
                </span>
              )}
              <span className="font-mono text-stone-400">{c.id}</span>
              <span className="font-medium text-stone-700">{c.author}</span>
              <span className="rounded bg-stone-100 px-1 text-[10px] uppercase tracking-wide text-stone-600">
                {c.kind}
              </span>
              {c.in_reply_to && (
                <span className="font-mono text-stone-400">→ {c.in_reply_to}</span>
              )}
              <time className="ml-auto tabular-nums">{compactTs(c.ts)}</time>
              {onDelete && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm("Delete this comment?")) {
                      onDelete(c.id);
                    }
                  }}
                  className="rounded p-0.5 text-stone-400 hover:bg-red-100 hover:text-red-700"
                  aria-label="Delete comment"
                  title="Delete comment"
                >
                  <X className="h-3 w-3" />
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
                {c.body}
              </ReactMarkdown>
            </div>
            {addressed && (
              <div
                className="mt-1.5 text-[10px] italic text-emerald-700"
                title={addressed.note || undefined}
              >
                addressed by Claude
                {addressed.sha && (
                  <>
                    {" · "}
                    <span className="font-mono">{addressed.sha}</span>
                  </>
                )}
              </div>
            )}
            {reply && reply.currentUserEmail && (
              <div className="mt-1.5 flex justify-end">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setReplyingTo((cur) => (cur === c.id ? null : c.id));
                  }}
                  className="rounded px-1.5 py-0.5 text-[11px] font-medium text-stone-500 hover:bg-stone-100 hover:text-stone-800"
                >
                  {replyingTo === c.id ? "Cancel" : "Reply"}
                </button>
              </div>
            )}
            {reply && replyingTo === c.id && (
              <ReplyComposer
                taskId={reply.taskId}
                postReply={reply.postReply}
                parentId={c.id}
                onPosted={async () => {
                  setReplyingTo(null);
                  await reply.onPosted();
                }}
                onCancel={() => setReplyingTo(null)}
                onPendingStart={reply.onPendingStart}
              />
            )}
            <PendingThread
              placeholders={pendingByParent[c.id] ?? []}
              onDismiss={reply?.onPendingDismiss}
            />
            <ReplyThread
              parentId={c.id}
              repliesByParent={repliesByParent}
              hovered={hovered}
              setHovered={setHovered}
              setHoveredId={setHoveredId}
              isHighlighted={isHighlighted}
              onDelete={onDelete}
              depth={1}
              reply={reply}
              replyingTo={replyingTo}
              setReplyingTo={setReplyingTo}
              pendingByParent={pendingByParent}
            />
          </li>
        );
      })}
    </ul>
  );
}

/** Recursive nested-replies renderer. Each level indents and tints
 *  slightly so the threading is visually obvious. Replies don't
 *  participate in the top-level anchor-alignment math (no data-cid on
 *  the outer wrapper); the inner li still carries data-cid for hover.
 *
 *  When `reply` wiring is passed, each rendered row also gets a Reply
 *  button + inline composer + pending placeholder slot — exactly the
 *  same as the top-level rows, so a user can keep iterating with
 *  Claude N levels deep. We DON'T add extra depth-based indent here:
 *  the `border-l-2` + `pl-3` per level already communicates depth and
 *  scales without runaway nesting in narrow rails. */
function ReplyThread({
  parentId,
  repliesByParent,
  hovered,
  setHovered,
  setHoveredId,
  isHighlighted,
  onDelete,
  depth,
  reply,
  replyingTo,
  setReplyingTo,
  pendingByParent,
}: {
  parentId: string;
  repliesByParent: Record<string, TaskComment[]>;
  hovered: string | null;
  setHovered: (id: string | null) => void;
  setHoveredId: (id: string | null) => void;
  isHighlighted: (id: string) => boolean;
  onDelete?: (id: string) => void;
  depth: number;
  reply?: ReplyWiring;
  replyingTo?: string | null;
  setReplyingTo?: (id: string | null) => void;
  pendingByParent?: Record<string, PendingPlaceholder[]>;
}) {
  const replies = repliesByParent[parentId];
  if (!replies || replies.length === 0) return null;
  return (
    <ul className="mt-2 space-y-1.5 border-l-2 border-stone-200 pl-3">
      {replies.map((c) => {
        const highlighted = isHighlighted(c.id);
        const isClaudeReply = c.author === "claude" || c.kind === "anchor-comment-reply";
        return (
          <li
            key={c.id}
            data-cid={c.id}
            onMouseEnter={() => {
              setHovered(c.id);
              setHoveredId(c.id);
            }}
            onMouseLeave={() => {
              setHovered(null);
              setHoveredId(null);
            }}
            className={`rounded border px-2.5 py-2 text-sm transition-colors duration-100 ${
              highlighted
                ? "border-amber-300 bg-amber-50"
                : isClaudeReply
                  ? "border-stone-200 bg-stone-50"
                  : "border-stone-200 bg-white"
            }`}
          >
            <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-stone-500">
              <span className="font-medium text-stone-700">
                {isClaudeReply ? "Claude" : c.author}
              </span>
              <time className="ml-auto tabular-nums">{compactTs(c.ts)}</time>
              {onDelete && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm("Delete this reply?")) {
                      onDelete(c.id);
                    }
                  }}
                  className="rounded p-0.5 text-stone-400 hover:bg-red-100 hover:text-red-700"
                  aria-label="Delete reply"
                  title="Delete reply"
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>
            <div className="prose prose-sm prose-stone max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw, rehypeHighlight]}
              >
                {c.body}
              </ReactMarkdown>
            </div>
            {reply && reply.currentUserEmail && setReplyingTo && (
              <div className="mt-1.5 flex justify-end">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setReplyingTo(replyingTo === c.id ? null : c.id);
                  }}
                  className="rounded px-1.5 py-0.5 text-[11px] font-medium text-stone-500 hover:bg-stone-100 hover:text-stone-800"
                >
                  {replyingTo === c.id ? "Cancel" : "Reply"}
                </button>
              </div>
            )}
            {reply && setReplyingTo && replyingTo === c.id && (
              <ReplyComposer
                taskId={reply.taskId}
                postReply={reply.postReply}
                parentId={c.id}
                onPosted={async () => {
                  setReplyingTo(null);
                  await reply.onPosted();
                }}
                onCancel={() => setReplyingTo(null)}
                onPendingStart={reply.onPendingStart}
              />
            )}
            <PendingThread
              placeholders={pendingByParent?.[c.id] ?? []}
              onDismiss={reply?.onPendingDismiss}
            />
            {/* Recurse: replies-of-replies cascade down. Same wiring
                forwards so every nested row can host its own Reply
                composer + inline pending card. */}
            <ReplyThread
              parentId={c.id}
              repliesByParent={repliesByParent}
              hovered={hovered}
              setHovered={setHovered}
              setHoveredId={setHoveredId}
              isHighlighted={isHighlighted}
              onDelete={onDelete}
              depth={depth + 1}
              reply={reply}
              replyingTo={replyingTo}
              setReplyingTo={setReplyingTo}
              pendingByParent={pendingByParent}
            />
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Inline composer that opens beneath a comment when the user clicks
 * Reply. Default behavior: POSTs to /api/updates/comment with
 * `in_reply_to: parentId` so the server inherits the chain root's anchor
 * and routes the new row into the thread. Same Enter / Shift+Enter
 * semantics as the rail CardComposer.
 *
 * If `postReply` is provided, the composer calls THAT instead — used by
 * /log to route the reply to /api/log/comment (which keys off `entryId`
 * rather than `taskId`). The response shape is the same either way, so
 * the pending-placeholder + auto-refresh UX is identical.
 *
 * Lives in CommentList (not lifted into the page-level composer)
 * because each rendered row needs its own composer instance with its
 * own parentId. The wiring (taskId / postReply / onPosted /
 * onPendingStart) is forwarded by the caller; we don't import anything
 * from the /updates components tree.
 */
function ReplyComposer({
  taskId,
  postReply,
  parentId,
  onPosted,
  onCancel,
  onPendingStart,
}: {
  taskId?: number;
  postReply?: (parentId: string, body: string) => Promise<PostReplyResult>;
  parentId: string;
  onPosted: () => Promise<void>;
  onCancel: () => void;
  onPendingStart: (parentId: string, replyId: string) => void;
}) {
  const [draft, setDraft] = useState("");
  const [posting, setPosting] = useState(false);
  const [status, setStatus] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Autofocus the textarea on mount so clicking Reply puts the cursor
  // right where the user expects to type.
  useEffect(() => {
    taRef.current?.focus();
  }, []);

  async function submit() {
    if (!draft.trim() || posting) return;
    setPosting(true);
    setStatus(null);
    try {
      let json: PostReplyResult;
      if (postReply) {
        // Caller-supplied override (e.g. /log routes to /api/log/comment).
        json = await postReply(parentId, draft.trim());
      } else {
        // Default: tasks/updates path. Requires a taskId — refuse to
        // POST without one so we don't send `{taskId: undefined}` and
        // get a confusing server-side 400.
        if (typeof taskId !== "number") {
          setStatus({
            kind: "err",
            text: "ReplyComposer: missing taskId and no postReply override",
          });
          return;
        }
        const res = await fetch("/api/updates/comment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({ taskId, body: draft.trim(), in_reply_to: parentId }),
        });
        json = (await res.json()) as PostReplyResult;
      }
      if (!json.ok) {
        setStatus({ kind: "err", text: json.error });
        return;
      }
      setDraft("");
      // Pending placeholder is parented to the NEW user comment's id
      // so it nests under the just-posted row, not under the parent
      // we're replying to.
      if (json.will_reply && json.pending_reply_id) {
        onPendingStart(json.id, json.pending_reply_id);
      }
      await onPosted();
    } catch (e) {
      setStatus({ kind: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setPosting(false);
    }
  }

  return (
    <div className="mt-2 rounded border border-stone-300 bg-white p-2"
         onClick={(e) => e.stopPropagation()}>
      <textarea
        ref={taRef}
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          setStatus(null);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            void submit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            onCancel();
          }
        }}
        disabled={posting}
        placeholder="Reply (Enter to post, Shift+Enter for newline, Esc to cancel)"
        rows={2}
        className="w-full resize-y rounded border border-stone-300 bg-white px-2 py-1 text-sm font-mono"
      />
      <div className="mt-1 flex items-center gap-2">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            void submit();
          }}
          disabled={posting || !draft.trim()}
          className="rounded bg-stone-900 px-2 py-1 text-xs font-medium text-white disabled:bg-stone-300"
        >
          {posting ? "…" : "Post reply"}
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onCancel();
          }}
          className="rounded px-2 py-1 text-xs text-stone-600 hover:bg-stone-100"
        >
          Cancel
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

/**
 * Pending placeholder list for a single parent. Renders one card per
 * outstanding spinner. Shape mirrors PendingReplyCard in
 * CardCommentBox.tsx but the CommentList copy is intentionally
 * standalone so this file doesn't reach back into /components/updates.
 */
function PendingThread({
  placeholders,
  onDismiss,
}: {
  placeholders: PendingPlaceholder[];
  onDismiss?: (replyId: string) => void;
}) {
  if (placeholders.length === 0) return null;
  return (
    <ul className="mt-2 space-y-1.5 border-l-2 border-amber-200 pl-3">
      {placeholders.map((p) => (
        <PendingItem key={p.id} placeholder={p} onDismiss={onDismiss} />
      ))}
    </ul>
  );
}

function PendingItem({
  placeholder,
  onDismiss,
}: {
  placeholder: PendingPlaceholder;
  onDismiss?: (replyId: string) => void;
}) {
  // Lazy initializer keeps `Date.now()` out of the render body (purity
  // lint rule). The interval re-pulls the wall clock once a second so
  // the elapsed counter ticks even though `now` itself isn't otherwise
  // re-derived from props.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (placeholder.state !== "pending") return;
    const i = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(i);
  }, [placeholder.state]);
  if (placeholder.state === "error") {
    return (
      <li className="rounded border border-amber-300 bg-amber-50 px-2.5 py-2 text-xs text-amber-900">
        <div className="flex items-start justify-between gap-2">
          <span>Claude didn&rsquo;t respond — try again or check the server logs.</span>
          {onDismiss && (
            <button
              type="button"
              onClick={() => onDismiss(placeholder.id)}
              className="rounded p-0.5 text-amber-700 hover:bg-amber-100 hover:text-amber-900"
              aria-label="Dismiss"
              title="Dismiss"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </li>
    );
  }
  const elapsedS = Math.max(0, Math.floor((now - placeholder.startedAt) / 1000));
  return (
    <li className="rounded border border-amber-300 bg-amber-50 px-2.5 py-2 text-xs">
      <div className="flex items-center gap-2 text-amber-900">
        <Loader2 className="h-4 w-4 animate-spin" />
        <div className="flex-1">
          <div className="font-medium">Claude is working…</div>
          <div className="text-[10px] text-amber-700 tabular-nums">{elapsedS}s elapsed</div>
        </div>
      </div>
    </li>
  );
}

/** "2026-05-20T08:14:00Z" → "05-20 08:14". Drops year + seconds for compact display. */
function compactTs(ts: string): string {
  const m = ts.match(/^\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]}` : ts;
}

/** Pull anchor quote from a comment's optional extras (set by CommentForm). */
function readAnchorQuote(c: TaskComment): string | null {
  const a = (c as Record<string, unknown>).anchor;
  if (a && typeof a === "object" && a !== null) {
    const q = (a as { quote?: unknown }).quote;
    if (typeof q === "string" && q.trim()) return q;
  }
  return null;
}

/**
 * Read the addressed marker set by /api/updates/address-comments. We
 * stash three optional fields on the row (`addressed`, `addressed_in`,
 * `addressed_note`) so the dashboard can render a small "addressed by
 * Claude · <sha>" badge without changing the TaskComment shape.
 */
function readAddressed(
  c: TaskComment,
): { sha: string | null; note: string | null } | null {
  const raw = c as Record<string, unknown>;
  if (raw.addressed !== true) return null;
  const sha = typeof raw.addressed_in === "string" ? raw.addressed_in : null;
  const note = typeof raw.addressed_note === "string" ? raw.addressed_note : null;
  return { sha, note };
}
