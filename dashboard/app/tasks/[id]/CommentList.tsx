"use client";

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
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
 *     each render to align without flicker.
 *   - Hovering a comment highlights two related sets:
 *       (a) the comment it replies to (`in_reply_to`)
 *       (b) any comments that reply TO this comment
 *     and also tells the body (via context) to darken the matching <mark>.
 *   - Clicking an anchored comment scrolls its <mark> into view.
 */
export function CommentList({ comments }: { comments: TaskComment[] }) {
  const [hovered, setHovered] = useState<string | null>(null);
  const { setHoveredId, requestScrollTo, anchorPositions } = useAnchoredComments();
  const listRef = useRef<HTMLUListElement>(null);

  const anchorTopById = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of anchorPositions) m.set(p.id, p.top);
    return m;
  }, [anchorPositions]);

  // Sort: anchored first by anchor Y (= text order), then unanchored by ts.
  const sorted = useMemo(() => {
    const list = [...comments];
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
    if (!hovered) return false;
    if (cid === hovered) return true;
    const hoveredC = byId[hovered];
    if (hoveredC?.in_reply_to === cid) return true;
    if ((childrenOf[hovered] || []).includes(cid)) return true;
    return false;
  }

  // Vertical alignment: push each anchored comment down until its rendered
  // top matches its anchor's document-coords top. Runs after every layout
  // pass (useLayoutEffect so the user never sees the un-aligned frame).
  useLayoutEffect(() => {
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
  }, [sorted, anchorTopById]);

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
              <span className="font-mono text-stone-400">{c.id}</span>
              <span className="font-medium text-stone-700">{c.author}</span>
              <span className="rounded bg-stone-100 px-1 text-[10px] uppercase tracking-wide text-stone-600">
                {c.kind}
              </span>
              {c.in_reply_to && (
                <span className="font-mono text-stone-400">→ {c.in_reply_to}</span>
              )}
              <time className="ml-auto tabular-nums">{compactTs(c.ts)}</time>
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
