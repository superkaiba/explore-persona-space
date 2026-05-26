"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { useAnchoredComments, type AnchorPosition } from "./AnchoredCommentsContext";

const COMMITTED_BG = "rgb(254 243 199)"; // amber-100
const COMMITTED_BG_HOVER = "rgb(253 230 138)"; // amber-200
const PENDING_BG = "rgba(252, 211, 77, 0.35)"; // amber-300 @35%

/**
 * Body of a task wrapped in selection capture + anchor highlighting.
 *
 * Sagan-style behavior:
 *   - Drag-select text inside the body → "+ Comment on selection" popover
 *     appears next to the selection.
 *   - Click → the selection is wrapped in `<mark data-anchor-pending>` so
 *     the highlight stays visible while you type in the sidebar. Pending
 *     mark is auto-removed when the comment is posted (pendingQuote
 *     cleared by CommentForm).
 *   - Once posted, the comment's anchor.quote is wrapped in
 *     `<mark data-comment-id>` everywhere it occurs in the body.
 *   - Hovering a comment in the sidebar darkens ALL matching marks +
 *     adds an amber ring.
 *   - Clicking a comment in the sidebar scrolls its FIRST mark into view
 *     (smooth, centered).
 */
export function CommentableBody({
  body,
  isLegacyHtml,
}: {
  body: string;
  isLegacyHtml: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const pendingRangeRef = useRef<Range | null>(null);
  const {
    anchors,
    hoveredId,
    setHoveredId,
    pendingQuote,
    setPendingQuote,
    setAnchorPositions,
    scrollToCommentId,
    clearScrollRequest,
  } = useAnchoredComments();
  const [popover, setPopover] = useState<
    { top: number; left: number; quote: string } | null
  >(null);

  // --- Selection capture --------------------------------------------------
  useEffect(() => {
    function onMouseUp(e: MouseEvent) {
      const target = e.target as Node | null;
      if (!ref.current || !target) return;
      if (!ref.current.contains(target)) {
        setPopover(null);
        return;
      }
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
        setPopover(null);
        return;
      }
      const text = sel.toString().trim();
      if (text.length < 4) {
        setPopover(null);
        return;
      }
      const range = sel.getRangeAt(0);
      if (!ref.current.contains(range.commonAncestorContainer)) {
        setPopover(null);
        return;
      }
      const rect = range.getBoundingClientRect();
      const containerRect = ref.current.getBoundingClientRect();
      pendingRangeRef.current = range.cloneRange();
      setPopover({
        top: rect.bottom - containerRect.top + 6,
        left: Math.max(0, rect.left - containerRect.left),
        quote: text,
      });
    }
    document.addEventListener("mouseup", onMouseUp);
    return () => document.removeEventListener("mouseup", onMouseUp);
  }, []);

  // --- Pending-mark wrapping (so selection stays visible) ----------------
  function onPopoverClick() {
    if (!popover || !ref.current) return;
    const range = pendingRangeRef.current;
    if (range && ref.current.contains(range.commonAncestorContainer)) {
      // Remove any existing pending mark first (one at a time).
      unwrapMatching(ref.current, "mark[data-anchor-pending]");
      wrapRange(range, { pending: true });
    }
    setPendingQuote(popover.quote);
    setPopover(null);
    window.getSelection()?.removeAllRanges();
    pendingRangeRef.current = null;
  }

  // Tear down the pending mark when CommentForm clears pendingQuote
  // (post succeeded, or user clicked the ✕).
  useEffect(() => {
    if (!ref.current) return;
    if (pendingQuote === null) {
      unwrapMatching(ref.current, "mark[data-anchor-pending]");
    }
  }, [pendingQuote]);

  // --- Committed-anchor wrapping + position publish ----------------------
  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    unwrapMatching(root, "mark[data-comment-id]");
    for (const a of anchors) {
      wrapAllOccurrences(root, a.quote, a.id);
    }
    publishPositions(root, setAnchorPositions);

    // Re-measure on resize / layout shifts (font load, image load, etc).
    const onResize = () => publishPositions(root, setAnchorPositions);
    window.addEventListener("resize", onResize);
    const ro = new ResizeObserver(() => publishPositions(root, setAnchorPositions));
    ro.observe(root);
    // Also re-measure after a short delay to catch async font/image
    // settling that ResizeObserver might miss on the very first paint.
    const t = window.setTimeout(onResize, 100);
    return () => {
      window.removeEventListener("resize", onResize);
      ro.disconnect();
      window.clearTimeout(t);
    };
  }, [anchors, body, setAnchorPositions]);

  // --- Hover sync ---------------------------------------------------------
  // Comment-list-hover → tint the matching marks in the body. Driven by
  // the shared `hoveredId` from the context, which the CommentList sets
  // on mouseenter.
  useEffect(() => {
    if (!ref.current) return;
    ref.current
      .querySelectorAll<HTMLElement>("mark[data-comment-id]")
      .forEach((m) => {
        const active = m.dataset.commentId === hoveredId;
        m.style.background = active ? COMMITTED_BG_HOVER : COMMITTED_BG;
        if (active) m.classList.add("ring-2", "ring-amber-400");
        else m.classList.remove("ring-2", "ring-amber-400");
      });
  }, [hoveredId, anchors]);

  // Reverse direction: body-mark-hover → set `hoveredId` so the matching
  // comment in the list highlights too. Attach mouse listeners to every
  // `<mark data-comment-id>` whenever the rendered marks change.
  useEffect(() => {
    if (!ref.current) return;
    const marks = ref.current.querySelectorAll<HTMLElement>("mark[data-comment-id]");
    const handlers: Array<() => void> = [];
    marks.forEach((m) => {
      const id = m.dataset.commentId;
      if (!id) return;
      const onEnter = () => setHoveredId(id);
      const onLeave = () => setHoveredId(null);
      m.addEventListener("mouseenter", onEnter);
      m.addEventListener("mouseleave", onLeave);
      handlers.push(() => {
        m.removeEventListener("mouseenter", onEnter);
        m.removeEventListener("mouseleave", onLeave);
      });
    });
    return () => {
      for (const off of handlers) off();
    };
  }, [anchors, setHoveredId]);

  // --- Scroll-to-mark (sidebar click) ------------------------------------
  useEffect(() => {
    if (!ref.current || !scrollToCommentId) return;
    const mark = ref.current.querySelector<HTMLElement>(
      `mark[data-comment-id="${cssEscape(scrollToCommentId)}"]`,
    );
    if (mark) {
      mark.scrollIntoView({ block: "center", behavior: "smooth" });
    }
    clearScrollRequest();
  }, [scrollToCommentId, clearScrollRequest]);

  return (
    <div ref={ref} className="relative">
      <div className="prose prose-sm sm:prose-base prose-stone max-w-none">
        {isLegacyHtml ? (
          <div
            className="legacy-sagan-card"
            // Legacy Sagan-card bodies are trusted HTML.
            dangerouslySetInnerHTML={{ __html: body }}
          />
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw, rehypeHighlight]}
          >
            {body}
          </ReactMarkdown>
        )}
      </div>
      {popover && (
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault() /* keep the selection alive */}
          onClick={onPopoverClick}
          style={{ position: "absolute", top: popover.top, left: popover.left }}
          className="z-10 rounded bg-stone-900 px-2 py-1 text-xs font-medium text-white shadow hover:bg-stone-700"
        >
          + Comment on selection
        </button>
      )}
    </div>
  );
}

// ─── DOM helpers ──────────────────────────────────────────────────────────

/**
 * Measure each committed mark's top (in document coords) and publish the
 * positions via context. CommentList uses these to sort + vertically align
 * its comments next to their anchors.
 *
 * If a comment has multiple matches in the body, we use the FIRST one's
 * position (same first-match the scroll-to-mark behavior uses).
 */
function publishPositions(
  root: HTMLElement,
  setAnchorPositions: (positions: AnchorPosition[]) => void,
) {
  const seen = new Map<string, AnchorPosition>();
  root.querySelectorAll<HTMLElement>("mark[data-comment-id]").forEach((m) => {
    const id = m.dataset.commentId;
    if (!id || seen.has(id)) return;
    const rect = m.getBoundingClientRect();
    seen.set(id, {
      id,
      top: rect.top + window.scrollY,
      height: rect.height,
    });
  });
  setAnchorPositions(Array.from(seen.values()));
}

/** Remove all marks matching the selector, restoring their children in place. */
function unwrapMatching(root: HTMLElement, selector: string) {
  root.querySelectorAll(selector).forEach((m) => {
    const p = m.parentNode;
    if (!p) return;
    while (m.firstChild) p.insertBefore(m.firstChild, m);
    p.removeChild(m);
  });
  root.normalize();
}

/**
 * Wrap an arbitrary Range in a fresh <mark>.
 *   - pending=true → dashed amber outline, no data-comment-id, marked
 *     with data-anchor-pending so we can find/unwrap it later.
 *   - pending=false → solid amber bg, data-comment-id=<id>.
 *
 * Uses extractContents + insertNode so it works even when the range
 * crosses element boundaries (the simple range.surroundContents API
 * throws on cross-element ranges).
 */
function wrapRange(
  range: Range,
  opts: { pending: true } | { pending: false; id: string },
) {
  const mark = document.createElement("mark");
  mark.className =
    "rounded px-0.5 transition-colors duration-100" +
    (opts.pending ? " outline-dashed outline-2 outline-amber-400" : "");
  mark.style.background = opts.pending ? PENDING_BG : COMMITTED_BG;
  if (opts.pending) {
    mark.dataset.anchorPending = "true";
  } else {
    mark.dataset.commentId = opts.id;
  }
  try {
    range.surroundContents(mark);
  } catch (_) {
    // Cross-element range: extract then reinsert.
    const frag = range.extractContents();
    mark.appendChild(frag);
    range.insertNode(mark);
  }
}

/**
 * Walk text nodes (skipping those inside existing marks), build a flat
 * text index, find ALL non-overlapping occurrences of `quote`, and wrap
 * each one in <mark data-comment-id="id">.
 *
 * Iterates the matches in reverse so earlier text-positions remain
 * valid for the later iterations (extractContents only affects the
 * region being wrapped — text before it is untouched).
 */
function wrapAllOccurrences(root: HTMLElement, quote: string, id: string) {
  const needle = quote.trim();
  if (needle.length < 4) return;

  const segments = collectTextSegments(root);
  if (segments.length === 0) return;
  const fullText = segments.map((s) => s.node.nodeValue ?? "").join("");

  const matches: Array<{ start: number; end: number }> = [];
  let idx = 0;
  while (true) {
    const found = fullText.indexOf(needle, idx);
    if (found === -1) break;
    matches.push({ start: found, end: found + needle.length });
    idx = found + needle.length;
  }
  if (matches.length === 0) return;

  // Reverse so each wrap doesn't shift the offsets of earlier matches.
  for (const m of matches.reverse()) {
    const startSeg = findSegment(segments, m.start);
    const endSeg = findSegment(segments, m.end - 1);
    if (!startSeg || !endSeg) continue;
    if (!isInDocument(startSeg.node) || !isInDocument(endSeg.node)) continue;
    const range = document.createRange();
    try {
      range.setStart(startSeg.node, m.start - startSeg.start);
      range.setEnd(endSeg.node, m.end - endSeg.start);
    } catch (_) {
      continue;
    }
    wrapRange(range, { pending: false, id });
  }
}

type Segment = { node: Text; start: number; end: number };

function collectTextSegments(root: HTMLElement): Segment[] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      // Skip text already inside a mark we manage.
      let p = node.parentElement;
      while (p && p !== root) {
        if (
          p.tagName === "MARK" &&
          (p.dataset.commentId !== undefined ||
            p.dataset.anchorPending !== undefined)
        ) {
          return NodeFilter.FILTER_REJECT;
        }
        p = p.parentElement;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const out: Segment[] = [];
  let pos = 0;
  let n = walker.nextNode() as Text | null;
  while (n) {
    const len = (n.nodeValue ?? "").length;
    out.push({ node: n, start: pos, end: pos + len });
    pos += len;
    n = walker.nextNode() as Text | null;
  }
  return out;
}

function findSegment(segments: Segment[], pos: number): Segment | undefined {
  // Binary search would be nicer; linear is fine for the sizes we see.
  for (const s of segments) {
    if (s.start <= pos && pos < s.end) return s;
  }
  return undefined;
}

function isInDocument(n: Node): boolean {
  return document.contains(n);
}

function cssEscape(s: string): string {
  if (typeof window !== "undefined" && (window as { CSS?: { escape?: typeof CSS.escape } }).CSS?.escape) {
    return CSS.escape(s);
  }
  return s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}
