"use client";

/**
 * <MarkdownDoc> — the shared markdown rendering keystone.
 *
 * Generalized from app/tasks/[id]/CommentableBody.tsx (which is now a thin
 * wrapper around this). ONE component backs every markdown surface in the
 * dashboard (tasks, docs, results, overview, updates). Capabilities, each
 * opt-in via a prop:
 *
 *   - SANITIZED render pipeline (always on). Markdown path:
 *       remarkGfm, remarkMath -> rehypeRaw -> rehypeSanitize(markdownSchema)
 *       -> rehypeKatex -> rehypeHighlight.
 *     Sanitize runs AFTER rehypeRaw (cleans injected raw HTML) and BEFORE the
 *     trusted class-adding plugins (katex/highlight) so their classes survive.
 *     Legacy Sagan-card bodies (isLegacyHtml) keep the dangerouslySetInnerHTML
 *     path but pass through `sanitizeLegacyHtml` first (these are now public).
 *
 *     Heading ids are NOT assigned by rehype-slug. A single client effect
 *     (`assignHeadingIds`) assigns them with the SAME `githubLikeSlug` +
 *     `dedupeSlug` helpers the TOC and the collapsible layer use, namespaced
 *     per `docId`, on EVERY render path (collapsible or not). This is the one
 *     canonical slugger — using rehype-slug (github-slugger) in parallel would
 *     produce divergent ids for headings with stripped punctuation flanked by
 *     spaces (e.g. `p < 0.05`), so the TOC's `#slug` anchors would miss the
 *     rendered heading ids.
 *   - Per-header collapse (enableCollapsibleSections).
 *   - Auto table-of-contents rail (showToc).
 *   - Highlight-to-comment anchoring (default; disabled writes in public mode).
 *   - Ask-Claude affordance (enableAskClaude; disabled in public mode).
 *
 * Server/client boundary: this is a CLIENT component. Selection capture,
 * <mark> wrapping, collapse, TOC active-state, and hover-sync are all
 * post-hydration client effects over the rendered DOM — the server and
 * client both render the SAME sanitized markdown HTML, and marks are injected
 * only after mount, which avoids hydration mismatch. Callers pass `body` /
 * `comments` as serializable props; no disk reads or auth live here.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { PluggableList } from "unified";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import {
  useAnchoredComments,
  type AnchorPosition,
} from "@/app/tasks/[id]/AnchoredCommentsContext";
import {
  headingId as computeHeadingId,
  plainMarkdownText,
} from "@/lib/markdown-headings";
import { markdownSchema } from "@/lib/markdown-sanitize";
import { sanitizeLegacyHtml } from "@/lib/sanitize-legacy-html";
import { MarkdownDocToc } from "@/components/MarkdownDocToc";
import { MarkdownDocAskClaude } from "@/components/MarkdownDocAskClaude";

const COMMITTED_BG = "rgb(254 243 199)"; // amber-100
const COMMITTED_BG_HOVER = "rgb(253 230 138)"; // amber-200
const PENDING_BG = "rgba(252, 211, 77, 0.35)"; // amber-300 @35%

// Inline composer card width (px). Kept in sync with the clamp helper so the
// card never overflows the prose container on narrow viewports.
const COMPOSER_WIDTH = 288; // 18rem

const REMARK_PLUGINS: PluggableList = [remarkGfm, remarkMath];
// Order matters: raw -> sanitize -> (trusted plugins that add classes).
// Sanitize must run AFTER rehypeRaw (so injected raw HTML is cleaned) and
// BEFORE katex/highlight (so the classes those add survive).
//
// Heading ids are deliberately NOT assigned here (no rehype-slug). They are
// assigned by the `assignHeadingIds` client effect below using the project's
// own `githubLikeSlug` + `dedupeSlug` so the TOC, the rendered heading ids,
// and the collapsible-section ids all come from ONE slugger and agree on
// every path (rehype-slug's github-slugger diverges on `p < 0.05`-style
// headings). The collapsible layer reuses whatever ids that effect assigned.
const REHYPE_PLUGINS: PluggableList = [
  rehypeRaw,
  [rehypeSanitize, markdownSchema],
  rehypeKatex,
  rehypeHighlight,
];

export type MarkdownDocProps = {
  /** Markdown source, OR trusted legacy HTML when `isLegacyHtml`. */
  body: string;
  /**
   * When true, `body` is a trusted legacy Sagan-card HTML fragment rendered
   * via dangerouslySetInnerHTML (after sanitization). When false (default),
   * `body` is markdown rendered through the full plugin pipeline.
   */
  isLegacyHtml?: boolean;
  /**
   * When true, H1/H2/H3 headings become click-to-collapse. Section content is
   * the run of following siblings up to (but not including) the next heading
   * at the same-or-shallower depth. Default off.
   *
   * Collapse state persists per (docId, headingId) in localStorage when
   * `docId` is set.
   */
  enableCollapsibleSections?: boolean;
  /**
   * Stable id used to (a) namespace localStorage collapse state and (b) scope
   * the `eps:section-expand` event so multiple MarkdownDocs on a page don't
   * cross-toggle. For task pages this is the task id (number); any string is
   * fine for other surfaces.
   */
  docId?: string | number;
  /** Render a sticky left-rail table of contents (H1/H2/H3). Default off. */
  showToc?: boolean;
  /**
   * Mount an Ask-Claude affordance (a button that opens the global sidecar
   * chat panel with this doc's text as context). Default off. Rendered
   * DISABLED when `public` is true.
   */
  enableAskClaude?: boolean;
  /** Title shown in the Ask-Claude panel header / context. */
  askClaudeTitle?: string;
  /**
   * Public/read-only mode. When true: the "+ Comment on selection" popover
   * and all comment writes are DISABLED, and Ask-Claude renders DISABLED with
   * no `/api/chat-token` fetch. Sanitize applies regardless of this flag.
   *
   * Note on marks: this flag only gates the WRITE affordances. Whether any
   * committed `<mark>` anchors render depends on whether an
   * <AnchoredCommentsProvider> is mounted above this component — the public
   * surfaces (/, /results/[id]) do NOT mount one, so `useAnchoredComments`
   * falls back to an empty anchor list and no marks render there at all. (The
   * /docs/[slug] surface DOES mount a provider and is not in `public` mode, so
   * its marks render and are editable.)
   */
  public?: boolean;
  /**
   * Inline-composer hook. When provided AND writes are enabled (not `public`),
   * clicking "+ Comment on selection" opens a SMALL inline composer anchored
   * at the selection rect (a card with a textarea + Comment/Cancel) instead of
   * only setting `pendingQuote`. On Comment, the composer calls
   * `onCommentCreate({ quote, body })`; on a `true` return it clears itself +
   * the pending mark (the parent is expected to refetch anchors). The composer
   * is positioned absolutely near the selection and clamps into the viewport,
   * so highlight-to-comment works at any screen width with NO dependence on a
   * side rail.
   *
   * When NOT provided, the legacy `pendingQuote` flow is kept intact: the
   * popover click sets `pendingQuote` in context and a far-away rail composer
   * reads it. This keeps any caller relying on the old rail working.
   */
  onCommentCreate?: (args: { quote: string; body: string }) => Promise<boolean>;
  /** Extra classes on the prose container. */
  className?: string;
};

export function MarkdownDoc({
  body,
  isLegacyHtml = false,
  enableCollapsibleSections = false,
  docId,
  showToc = false,
  enableAskClaude = false,
  askClaudeTitle,
  public: isPublic = false,
  onCommentCreate,
  className,
}: MarkdownDocProps) {
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
  // Inline composer state (only used when `onCommentCreate` is provided).
  const [composer, setComposer] = useState<
    { top: number; left: number; quote: string } | null
  >(null);
  const [composerDraft, setComposerDraft] = useState("");
  const [composerBusy, setComposerBusy] = useState(false);
  const [composerError, setComposerError] = useState<string | null>(null);

  // Comment writes are disabled in public/read-only mode.
  const commentsEnabled = !isPublic;
  // When a create hook is wired AND writes are enabled, the popover opens the
  // inline composer instead of the legacy pendingQuote rail flow.
  const inlineComposerEnabled = commentsEnabled && typeof onCommentCreate === "function";

  // Stringify docId for storage keys / event scoping.
  const docKey = docId == null ? undefined : String(docId);

  // Sanitize legacy HTML once per body change.
  const legacyHtml = useMemo(
    () => (isLegacyHtml ? sanitizeLegacyHtml(body) : null),
    [isLegacyHtml, body],
  );

  // --- Heading id assignment (the canonical slugger, every path) ----------
  // rehype-slug is intentionally NOT in the pipeline. We assign heading ids
  // here so the TOC entries, the rendered heading ids, and the collapsible-
  // section ids all come from ONE slugger (githubLikeSlug + dedupeSlug + a
  // per-doc prefix) and therefore always agree — including for headings with
  // stripped punctuation flanked by spaces (`p < 0.05`, `A & B: results`)
  // where github-slugger would emit a divergent id. Runs whether or not
  // collapsible sections are enabled; the collapsible layer reuses the ids
  // assigned here. Skipped for legacy HTML (its anchors come from the trusted
  // markup itself).
  useEffect(() => {
    if (isLegacyHtml || !ref.current) return;
    const root = ref.current.querySelector<HTMLElement>(".prose");
    if (!root) return;
    assignHeadingIds(root, docKey);
    // Re-run if the rendered tree changes (collapsible layer re-wraps
    // headings into <section>; katex/highlight mutate inner spans). The ids
    // are idempotent — an already-correct id is left untouched.
    const mo = new MutationObserver(() => assignHeadingIds(root, docKey));
    mo.observe(root, { childList: true, subtree: true });
    return () => mo.disconnect();
  }, [body, docKey, isLegacyHtml]);

  // --- Selection capture (gated by commentsEnabled) -----------------------
  useEffect(() => {
    if (!commentsEnabled) return;
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
  }, [commentsEnabled]);

  // --- Pending-mark wrapping (so selection stays visible) ----------------
  function onPopoverClick() {
    if (!popover || !ref.current) return;
    const range = pendingRangeRef.current;
    if (range && ref.current.contains(range.commonAncestorContainer)) {
      unwrapMatching(ref.current, "mark[data-anchor-pending]");
      wrapRange(range, { pending: true });
    }

    if (inlineComposerEnabled) {
      // Inline-composer path: keep the pending mark visible and open a small
      // composer card anchored at the same selection rect. NO rail dependence.
      setComposerDraft("");
      setComposerError(null);
      setComposer({ top: popover.top, left: popover.left, quote: popover.quote });
    } else {
      // Legacy rail path: hand the quote off to the side-rail composer.
      setPendingQuote(popover.quote);
    }

    setPopover(null);
    window.getSelection()?.removeAllRanges();
    pendingRangeRef.current = null;
  }

  // Inline composer: cancel + clear the pending mark.
  function closeComposer() {
    setComposer(null);
    setComposerDraft("");
    setComposerError(null);
    if (ref.current) unwrapMatching(ref.current, "mark[data-anchor-pending]");
  }

  // Inline composer: submit via the parent hook. On success the parent
  // refetches anchors (which re-wraps the committed <mark>); we just clear the
  // composer + the pending highlight.
  async function submitComposer() {
    if (!composer || !onCommentCreate) return;
    const text = composerDraft.trim();
    if (!text || composerBusy) return;
    setComposerBusy(true);
    setComposerError(null);
    try {
      const ok = await onCommentCreate({ quote: composer.quote, body: text });
      if (ok) {
        closeComposer();
      } else {
        setComposerError("Couldn't post comment.");
      }
    } catch {
      setComposerError("Network error.");
    } finally {
      setComposerBusy(false);
    }
  }

  // Tear down the pending mark when CommentForm clears pendingQuote.
  useEffect(() => {
    if (!ref.current) return;
    if (pendingQuote === null) {
      unwrapMatching(ref.current, "mark[data-anchor-pending]");
    }
  }, [pendingQuote]);

  // --- Collapsible H1/H2/H3 sections (opt-in via prop) -------------------
  useEffect(() => {
    if (!ref.current) return;
    if (isLegacyHtml || !enableCollapsibleSections) return;
    const root = ref.current.querySelector<HTMLElement>(".prose");
    if (!root) return;
    applyCollapsibleSections(root, docKey);

    const mo = new MutationObserver(() => {
      applyCollapsibleSections(root, docKey);
    });
    mo.observe(root, { childList: true });

    // TocSidebar / MarkdownDocToc emits this when the user clicks a TOC entry
    // — expand the matching section so the heading is visible after scroll.
    const onExpand = (e: Event) => {
      const detail = (e as CustomEvent<{ headingId: string; docId?: string }>).detail;
      if (!detail) return;
      // Scope: only react when the event targets this doc (or is unscoped).
      if (docKey != null && detail.docId != null && detail.docId !== docKey) {
        return;
      }
      const heading = root.querySelector<HTMLElement>(`#${cssEscape(detail.headingId)}`);
      if (!heading) return;
      const section = heading.closest<HTMLElement>("section[data-collapsible-section]");
      if (!section) return;
      setSectionCollapsed(section, false, docKey);
    };
    window.addEventListener("eps:section-expand", onExpand);
    return () => {
      mo.disconnect();
      window.removeEventListener("eps:section-expand", onExpand);
    };
  }, [body, enableCollapsibleSections, isLegacyHtml, docKey]);

  // --- Committed-anchor wrapping + position publish ----------------------
  useEffect(() => {
    if (!ref.current) return;
    const root = ref.current;
    unwrapMatching(root, "mark[data-comment-id]");
    for (const a of anchors) {
      wrapAllOccurrences(root, a.quote, a.id);
    }
    publishPositions(root, setAnchorPositions);

    const onResize = () => publishPositions(root, setAnchorPositions);
    window.addEventListener("resize", onResize);
    const ro = new ResizeObserver(() => publishPositions(root, setAnchorPositions));
    ro.observe(root);
    const t = window.setTimeout(onResize, 100);
    return () => {
      window.removeEventListener("resize", onResize);
      ro.disconnect();
      window.clearTimeout(t);
    };
  }, [anchors, body, setAnchorPositions]);

  // --- Hover sync (comment-list -> body) ---------------------------------
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

  // --- Hover sync (body -> comment-list) ---------------------------------
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

  const proseEl = (
    <div ref={ref} className="relative">
      {enableAskClaude && (
        <div className="mb-3 flex justify-end">
          <MarkdownDocAskClaude
            body={body}
            title={askClaudeTitle}
            docId={docKey}
            disabled={isPublic}
          />
        </div>
      )}
      <div className={"prose prose-sm sm:prose-base prose-stone max-w-none " + (className ?? "")}>
        {isLegacyHtml ? (
          <div
            className="legacy-sagan-card"
            // Legacy Sagan-card bodies are trusted analyzer HTML, but since
            // they are now rendered on public surfaces they are sanitized
            // (SVG + scoped style preserved; script/on*/javascript: stripped)
            // before reaching the DOM.
            dangerouslySetInnerHTML={{ __html: legacyHtml ?? "" }}
          />
        ) : (
          <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS}>
            {body}
          </ReactMarkdown>
        )}
      </div>
      {commentsEnabled && popover && !composer && (
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
      {inlineComposerEnabled && composer && (
        <div
          // Clamp the composer's left edge so its 18rem (288px) card stays
          // inside the prose container at any width. `top`/`left` are relative
          // to the prose container (same basis the popover uses).
          style={{
            position: "absolute",
            top: composer.top,
            left: clampComposerLeft(composer.left, ref.current),
            width: COMPOSER_WIDTH,
          }}
          // Keep selection-driven mousedown from bubbling to the document
          // mouseup handler that would otherwise re-evaluate the popover.
          onMouseDown={(e) => e.stopPropagation()}
          className="z-20 rounded-lg border border-stone-300 bg-white p-2.5 text-sm shadow-lg"
        >
          <div className="mb-1.5 line-clamp-2 border-l-2 border-amber-300 pl-2 text-[11px] italic text-stone-500">
            &ldquo;
            {composer.quote.length > 120
              ? composer.quote.slice(0, 120) + "…"
              : composer.quote}
            &rdquo;
          </div>
          <textarea
            autoFocus
            value={composerDraft}
            onChange={(e) => {
              setComposerDraft(e.target.value);
              setComposerError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                closeComposer();
              } else if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.metaKey &&
                !e.ctrlKey
              ) {
                e.preventDefault();
                void submitComposer();
              }
            }}
            rows={3}
            placeholder="Comment on this selection… (@claude to summon a reply)"
            className="w-full rounded border border-stone-300 px-2 py-1.5 text-sm text-stone-800 placeholder:text-stone-400"
          />
          <div className="mt-1.5 flex items-center justify-between gap-2">
            <span className="text-[11px] text-rose-600">{composerError}</span>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={closeComposer}
                className="rounded px-2 py-1 text-xs text-stone-500 hover:bg-stone-100 hover:text-stone-800"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void submitComposer()}
                disabled={!composerDraft.trim() || composerBusy}
                className="rounded bg-stone-900 px-3 py-1 text-xs font-medium text-white hover:bg-stone-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {composerBusy ? "Saving…" : "Comment"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  if (!showToc) return proseEl;

  // TOC rail layout. The container query keeps the rail next to the body when
  // wide and stacks it above when narrow (mirrors CardCommentBox's rail).
  return (
    <div className="@container">
      <div className="grid gap-6 @3xl:grid-cols-[200px_minmax(0,1fr)]">
        <MarkdownDocToc body={body} docId={docKey} rootRef={ref} />
        <div className="min-w-0">{proseEl}</div>
      </div>
    </div>
  );
}

// ─── DOM helpers ──────────────────────────────────────────────────────────
// (moved verbatim from the original CommentableBody so behavior is identical)

/**
 * Measure each committed mark's top (in document coords) and publish the
 * positions via context. CommentList uses these to sort + vertically align
 * its comments next to their anchors. First-match wins for multi-occurrence.
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
 *   - pending=true  -> dashed amber outline, data-anchor-pending.
 *   - pending=false -> solid amber bg, data-comment-id=<id>.
 * Uses extractContents + insertNode so it works across element boundaries.
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
  } catch {
    const frag = range.extractContents();
    mark.appendChild(frag);
    range.insertNode(mark);
  }
}

/**
 * Walk text nodes (skipping those inside existing marks), build a flat text
 * index, find ALL non-overlapping occurrences of `quote`, and wrap each one
 * in <mark data-comment-id="id">. Iterates matches in reverse so earlier
 * text-positions remain valid for later iterations.
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

  for (const m of matches.reverse()) {
    const startSeg = findSegment(segments, m.start);
    const endSeg = findSegment(segments, m.end - 1);
    if (!startSeg || !endSeg) continue;
    if (!isInDocument(startSeg.node) || !isInDocument(endSeg.node)) continue;
    const range = document.createRange();
    try {
      range.setStart(startSeg.node, m.start - startSeg.start);
      range.setEnd(endSeg.node, m.end - endSeg.start);
    } catch {
      continue;
    }
    wrapRange(range, { pending: false, id });
  }
}

type Segment = { node: Text; start: number; end: number };

function collectTextSegments(root: HTMLElement): Segment[] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
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
  for (const s of segments) {
    if (s.start <= pos && pos < s.end) return s;
  }
  return undefined;
}

function isInDocument(n: Node): boolean {
  return document.contains(n);
}

/**
 * Clamp the inline composer's left offset (relative to the prose container)
 * so its fixed-width card stays fully inside the container at any viewport
 * width. `left` is the raw selection-rect left (already container-relative).
 * Falls back to the raw value when the container isn't measurable yet.
 */
function clampComposerLeft(left: number, container: HTMLElement | null): number {
  if (!container) return Math.max(0, left);
  const max = Math.max(0, container.clientWidth - COMPOSER_WIDTH);
  return Math.min(Math.max(0, left), max);
}

function cssEscape(s: string): string {
  if (
    typeof window !== "undefined" &&
    (window as { CSS?: { escape?: typeof CSS.escape } }).CSS?.escape
  ) {
    return CSS.escape(s);
  }
  return s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

// ─── Heading id assignment (canonical slugger) ───────────────────────────

/**
 * Assign ids to every heading (H1-H6) under `root` using the project's
 * canonical slugger (githubLikeSlug + dedupeSlug + per-doc prefix). Walks ALL
 * heading ranks in document order — not just the H1-H3 the TOC shows — so the
 * dedupe `counts` advance over the same heading sequence `extractMarkdownHeadings`
 * walks (it also counts H1-H6 before filtering to H1-H3 for display). If we
 * counted only H1-H3 here, a body with e.g. an H4 `TL;DR` before an H2 `TL;DR`
 * would assign the H2 `tldr` while the TOC expected `tldr-2`, and the anchor
 * would miss. Counting all ranks keeps the suffix numbering identical.
 *
 * Idempotent: a heading whose id already equals the computed id is left
 * untouched, so the MutationObserver that re-runs this after the collapsible
 * layer wraps headings into <section> makes no further changes and the
 * observer settles.
 */
function assignHeadingIds(root: HTMLElement, docKey: string | undefined) {
  const counts = new Map<string, number>();
  const headings = root.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, h6");
  headings.forEach((heading) => {
    const text = plainMarkdownText(heading.textContent ?? "");
    if (!text) return;
    const id = computeHeadingId(text, docKey, counts);
    if (heading.id !== id) heading.id = id;
    heading.classList.add("scroll-mt-4");
  });
}

// ─── Collapsible H1/H2/H3 helpers ────────────────────────────────────────

const COLLAPSE_STORAGE_PREFIX = "eps:collapse:";

/**
 * Walk the rendered prose root, grouping each H1/H2/H3 + the run of following
 * siblings (up to the next heading at the SAME OR SHALLOWER depth) into a
 * <section data-collapsible-section>. Idempotent. Heading ids are assigned
 * using the same slug algorithm as the TOC.
 */
function applyCollapsibleSections(root: HTMLElement, docKey: string | undefined) {
  const counts = new Map<string, number>();
  const headings: HTMLElement[] = [];
  root.childNodes.forEach((n) => {
    if (!(n instanceof HTMLElement)) return;
    if (/^H[1-3]$/.test(n.tagName)) headings.push(n);
  });

  for (const heading of headings) {
    if (heading.parentElement?.matches("section[data-collapsible-section]")) continue;
    const depth = Number(heading.tagName.charAt(1));
    if (!(depth === 1 || depth === 2 || depth === 3)) continue;

    // Ids are normally assigned by the `assignHeadingIds` effect (the
    // canonical slugger) before this runs. If a heading somehow lacks one,
    // fall back to the SAME prefixed computation so it still lines up with
    // the TOC; otherwise just keep the count in sync for dedupe.
    if (!heading.id) {
      const text = plainMarkdownText(heading.textContent ?? "");
      heading.id = computeHeadingId(text, docKey, counts);
    } else {
      counts.set(heading.id, (counts.get(heading.id) ?? 0) + 1);
    }
    heading.classList.add("scroll-mt-4");

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

    const section = document.createElement("section");
    section.dataset.collapsibleSection = "true";
    section.dataset.headingId = heading.id;
    const contentWrap = document.createElement("div");
    contentWrap.dataset.sectionContent = "true";

    heading.parentNode?.insertBefore(section, heading);
    section.appendChild(heading);
    for (const node of sectionContent) {
      contentWrap.appendChild(node);
    }
    section.appendChild(contentWrap);

    const initiallyCollapsed = readCollapseState(docKey, heading.id);
    setSectionCollapsed(section, initiallyCollapsed, docKey);

    if (!(heading as HTMLElement & { __collapsibleWired?: boolean }).__collapsibleWired) {
      heading.style.cursor = "pointer";
      heading.addEventListener("click", (e) => {
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
        setSectionCollapsed(section, !collapsed, docKey);
      });
      (heading as HTMLElement & { __collapsibleWired?: boolean }).__collapsibleWired = true;
    }
  }
}

function setSectionCollapsed(
  section: HTMLElement,
  collapsed: boolean,
  docKey: string | undefined,
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
  const headingId = section.dataset.headingId;
  if (docKey != null && headingId) {
    try {
      const key = `${COLLAPSE_STORAGE_PREFIX}${docKey}:${headingId}`;
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

function readCollapseState(docKey: string | undefined, headingId: string): boolean {
  if (docKey == null) return false;
  try {
    const key = `${COLLAPSE_STORAGE_PREFIX}${docKey}:${headingId}`;
    return window.localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}
