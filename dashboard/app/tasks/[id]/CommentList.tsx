"use client";

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { Check, X } from "lucide-react";
import type { TaskComment } from "@/lib/tasks";
import { useAnchoredComments } from "./AnchoredCommentsContext";

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

  function isHighlighted(cid: string): boolean {
    const h = effectiveHovered;
    if (!h) return false;
    if (cid === h) return true;
    const hoveredC = byId[h];
    if (hoveredC?.in_reply_to === cid) return true;
    if ((childrenOf[h] || []).includes(cid)) return true;
    return false;
  }

  // Vertical alignment: push each anchored comment down until its rendered
  // top matches its anchor's document-coords top. Runs after every layout
  // pass (useLayoutEffect so the user never sees the un-aligned frame).
  // Skipped entirely in inline mode (the comments flow under the body
  // and don't need to align with anything in document coords).
  useLayoutEffect(() => {
    if (inline) return;
    const list = listRef.current;
    if (!list || anchorTopById.size === 0) return;
    const items = Array.from(list.querySelectorAll<HTMLLIElement>("li[data-cid]"));
    // Reset all margins first so we compute on a clean slate.
    for (const li of items) li.style.marginTop = "";
    if (items.length === 0) return;
    const listDocTop = list.getBoundingClientRect().top + window.scrollY;
    let prevBottom = listDocTop; // running floor for collision avoidance
    const GAP = 8; // px between stacked comments
    for (const li of items) {
      const cid = li.dataset.cid!;
      const liRect = li.getBoundingClientRect();
      const liDocTop = liRect.top + window.scrollY;
      const naturalHeight = liRect.height;
      const anchorTop = anchorTopById.get(cid);
      if (anchorTop === undefined) {
        // Unanchored: stack normally; just update the running floor.
        prevBottom = liDocTop + naturalHeight;
        continue;
      }
      const desiredTop = Math.max(anchorTop, prevBottom + GAP);
      const delta = desiredTop - liDocTop;
      if (delta > 0) {
        li.style.marginTop = `${delta}px`;
        prevBottom = desiredTop + naturalHeight;
      } else {
        prevBottom = liDocTop + naturalHeight;
      }
    }
  }, [sorted, anchorTopById, inline]);

  if (comments.length === 0) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-3 py-4 text-center text-xs text-stone-500">
        No comments yet.
      </p>
    );
  }

  return (
    <ul ref={listRef} className="space-y-1.5">
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
            <ReplyThread
              parentId={c.id}
              repliesByParent={repliesByParent}
              hovered={hovered}
              setHovered={setHovered}
              setHoveredId={setHoveredId}
              isHighlighted={isHighlighted}
              onDelete={onDelete}
              depth={1}
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
 *  the outer wrapper); the inner li still carries data-cid for hover. */
function ReplyThread({
  parentId,
  repliesByParent,
  hovered,
  setHovered,
  setHoveredId,
  isHighlighted,
  onDelete,
  depth,
}: {
  parentId: string;
  repliesByParent: Record<string, TaskComment[]>;
  hovered: string | null;
  setHovered: (id: string | null) => void;
  setHoveredId: (id: string | null) => void;
  isHighlighted: (id: string) => boolean;
  onDelete?: (id: string) => void;
  depth: number;
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
            {/* Recurse: replies-of-replies cascade down. */}
            <ReplyThread
              parentId={c.id}
              repliesByParent={repliesByParent}
              hovered={hovered}
              setHovered={setHovered}
              setHoveredId={setHoveredId}
              isHighlighted={isHighlighted}
              onDelete={onDelete}
              depth={depth + 1}
            />
          </li>
        );
      })}
    </ul>
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
