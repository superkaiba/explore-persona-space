"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { useAnchoredComments, type AnchorPosition } from "./AnchoredCommentsContext";
import {
  dedupeSlug,
  githubLikeSlug,
  plainMarkdownText,
} from "@/lib/markdown-headings";

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
  enableCollapsibleSections = false,
  taskId,
}: {
  body: string;
  isLegacyHtml: boolean;
  /**
   * When true, H1/H2/H3 headings become click-to-collapse. Section content
   * is the run of following siblings up to (but not including) the next
   * heading at the same-or-shallower depth. Default off so the
   * /tasks/[id] page and other small-card renderers stay unchanged.
   *
   * Collapse state persists per (taskId, headingId) in localStorage. If
   * taskId is missing the layer still works but state doesn't survive
   * page reloads.
   */
  enableCollapsibleSections?: boolean;
  taskId?: number;
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

  // --- Collapsible H1/H2/H3 sections (opt-in via prop) -------------------
  // Post-process the rendered ReactMarkdown DOM: for each H1/H2/H3, hoist
  // it + the run of following siblings (up to the next heading at the
  // SAME OR SHALLOWER depth) into a <section data-collapsible-section>
  // wrapper. The heading gets an injected chevron button that toggles the
  // wrapper's `data-collapsed` flag, which drives a CSS rule that hides
  // the content via display:none.
  //
  // Heading ids are assigned here too (same slug algorithm as the TOC
  // sidebar), so #anchor links and the TocSidebar's scrollIntoView both
  // resolve. Idempotent: re-runs whenever `body` changes.
  //
  // Why run BEFORE the anchor-wrap effect below: the section wrappers
  // are non-MARK elements, so wrapAllOccurrences still walks the text
  // unchanged. If a committed mark lands inside a collapsed section it
  // simply isn't visible until the user expands; comment row alignment
  // (publishPositions) returns top=0 for hidden marks and CommentList
  // handles that gracefully.
  useEffect(() => {
    if (!ref.current) return;
    if (isLegacyHtml || !enableCollapsibleSections) return;
    const root = ref.current.querySelector<HTMLElement>(".prose");
    if (!root) return;
    applyCollapsibleSections(root, taskId);

    // Re-apply if some downstream effect (e.g. anchor wrap) reshuffles
    // siblings — we listen for childList mutations at the prose root.
    // Anchor wrap only touches text-node descendants, so this should
    // rarely fire, but it keeps us robust against future changes.
    const mo = new MutationObserver(() => {
      // applyCollapsibleSections is idempotent: it skips headings that
      // already live inside a data-collapsible-section wrapper.
      applyCollapsibleSections(root, taskId);
    });
    mo.observe(root, { childList: true });

    // TocSidebar emits this when the user clicks a TOC entry — expand
    // the matching section so the heading is visible after scroll.
    const onExpand = (e: Event) => {
      const detail = (e as CustomEvent<{ headingId: string; taskId?: number }>).detail;
      if (!detail || (taskId != null && detail.taskId !== taskId)) return;
      const heading = root.querySelector<HTMLElement>(`#${cssEscape(detail.headingId)}`);
      if (!heading) return;
      const section = heading.closest<HTMLElement>("section[data-collapsible-section]");
      if (!section) return;
      setSectionCollapsed(section, false, taskId);
    };
    window.addEventListener("eps:section-expand", onExpand);
    return () => {
      mo.disconnect();
      window.removeEventListener("eps:section-expand", onExpand);
    };
  }, [body, enableCollapsibleSections, isLegacyHtml, taskId]);

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

// ─── Collapsible H1/H2/H3 helpers ────────────────────────────────────────

const COLLAPSE_STORAGE_PREFIX = "eps:collapse:";

/**
 * Walk the rendered prose root, grouping each H1/H2/H3 + the run of
 * following siblings (up to the next heading at the SAME OR SHALLOWER
 * depth) into a <section data-collapsible-section>. Idempotent: skips
 * headings that already live inside a wrapper. Heading ids are assigned
 * using the same slug algorithm as the TOC.
 */
function applyCollapsibleSections(root: HTMLElement, taskId: number | undefined) {
  const counts = new Map<string, number>();
  // Collect headings up-front because we'll be moving DOM nodes around.
  const headings: HTMLElement[] = [];
  root.childNodes.forEach((n) => {
    if (!(n instanceof HTMLElement)) return;
    if (/^H[1-3]$/.test(n.tagName)) headings.push(n);
  });

  for (const heading of headings) {
    // Skip if already wrapped.
    if (heading.parentElement?.matches("section[data-collapsible-section]")) continue;
    const depth = Number(heading.tagName.charAt(1));
    if (!(depth === 1 || depth === 2 || depth === 3)) continue;

    // Assign id if missing (or re-use existing one).
    if (!heading.id) {
      const text = plainMarkdownText(heading.textContent ?? "");
      heading.id = dedupeSlug(githubLikeSlug(text), counts);
    } else {
      // Still track the count so later same-text headings get -2 etc.
      counts.set(heading.id, (counts.get(heading.id) ?? 0) + 1);
    }
    heading.classList.add("scroll-mt-4");

    // Inject chevron button (idempotent).
    if (!heading.querySelector("button[data-collapsible-toggle]")) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.collapsibleToggle = "true";
      btn.setAttribute("aria-label", "Collapse section");
      btn.className =
        "mr-2 inline-flex h-5 w-5 items-center justify-center rounded text-stone-500 hover:bg-stone-200 hover:text-stone-900 align-middle transition-transform";
      btn.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>';
      heading.insertBefore(btn, heading.firstChild);
    }

    // Collect siblings until the next heading at depth ≤ current.
    const sectionContent: Node[] = [];
    let sib = heading.nextSibling;
    while (sib) {
      const next = sib.nextSibling;
      if (sib instanceof HTMLElement && /^H[1-3]$/.test(sib.tagName)) {
        const sibDepth = Number(sib.tagName.charAt(1));
        if (sibDepth <= depth) break;
      }
      sectionContent.push(sib);
      sib = next;
    }

    // Build wrapper structure:
    //   <section data-collapsible-section data-heading-id=ID data-collapsed=...>
    //     <heading/>
    //     <div data-section-content>...siblings...</div>
    //   </section>
    const section = document.createElement("section");
    section.dataset.collapsibleSection = "true";
    section.dataset.headingId = heading.id;
    const contentWrap = document.createElement("div");
    contentWrap.dataset.sectionContent = "true";

    // Replace heading in place with the wrapper, then move heading + siblings in.
    heading.parentNode?.insertBefore(section, heading);
    section.appendChild(heading);
    for (const node of sectionContent) {
      contentWrap.appendChild(node);
    }
    section.appendChild(contentWrap);

    // Determine initial collapsed state from localStorage.
    const initiallyCollapsed = readCollapseState(taskId, heading.id);
    setSectionCollapsed(section, initiallyCollapsed, taskId);

    // Wire toggle button. Use a single click handler on the heading
    // (chevron is purely decorative — clicking anywhere on the heading
    // toggles, which matches familiar collapsible-section UX).
    if (!(heading as HTMLElement & { __collapsibleWired?: boolean }).__collapsibleWired) {
      heading.style.cursor = "pointer";
      heading.addEventListener("click", (e) => {
        // Don't toggle when clicking interactive elements inside the heading
        // (e.g. links). The chevron button itself is fine — it's the click
        // target we want.
        const target = e.target as HTMLElement | null;
        if (
          target &&
          target !== heading &&
          target.closest("a, code") &&
          !target.closest("button[data-collapsible-toggle]")
        ) {
          return;
        }
        const collapsed = section.dataset.collapsed === "true";
        setSectionCollapsed(section, !collapsed, taskId);
      });
      (heading as HTMLElement & { __collapsibleWired?: boolean }).__collapsibleWired = true;
    }
  }
}

function setSectionCollapsed(
  section: HTMLElement,
  collapsed: boolean,
  taskId: number | undefined,
) {
  section.dataset.collapsed = collapsed ? "true" : "false";
  const content = section.querySelector<HTMLElement>("div[data-section-content]");
  if (content) {
    content.style.display = collapsed ? "none" : "";
  }
  const chevron = section.querySelector<HTMLElement>("button[data-collapsible-toggle]");
  if (chevron) {
    chevron.style.transform = collapsed ? "rotate(-90deg)" : "";
    chevron.setAttribute("aria-label", collapsed ? "Expand section" : "Collapse section");
  }
  // Persist state per (taskId, headingId).
  const headingId = section.dataset.headingId;
  if (taskId != null && headingId) {
    try {
      const key = `${COLLAPSE_STORAGE_PREFIX}${taskId}:${headingId}`;
      if (collapsed) {
        window.localStorage.setItem(key, "1");
      } else {
        window.localStorage.removeItem(key);
      }
    } catch {
      // localStorage unavailable / quota — silently ignore.
    }
  }
}

function readCollapseState(taskId: number | undefined, headingId: string): boolean {
  if (taskId == null) return false;
  try {
    const key = `${COLLAPSE_STORAGE_PREFIX}${taskId}:${headingId}`;
    return window.localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}
