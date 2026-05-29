"use client";

/**
 * Auto table-of-contents rail for <MarkdownDoc> (showToc).
 *
 * Generalizes two existing sidebars:
 *   - components/updates/TocSidebar.tsx — markdown-heading extraction +
 *     `eps:section-expand` to open a collapsed section on click.
 *   - components/tasks/TaskTocSidebar.tsx — IntersectionObserver
 *     active-highlight that follows the scroll position.
 *
 * Headings come from `extractMarkdownHeadings(body, docId)` (H1/H2/H3). On
 * click we (1) emit `eps:section-expand` (scoped to this doc via `docId`) so
 * the matching collapsible section opens if collapsed, then (2) smooth-scroll
 * the heading into view. Active highlight uses an IntersectionObserver over
 * the rendered heading ids; the topmost heading in the viewport's upper region
 * wins.
 *
 * Slug agreement: heading ids are assigned by MarkdownDoc's `assignHeadingIds`
 * effect using the SAME `headingId` (githubLikeSlug + dedupeSlug + per-doc
 * prefix) this TOC passes to `extractMarkdownHeadings`, so the `#<id>` anchors
 * line up on every render path.
 *
 * DOM scope: this TOC can be one of several on a page (the Overview renders
 * two docs, each with its own TOC). All heading lookups resolve WITHIN this
 * doc's own subtree (`rootRef`) — never `document`-global — so a heading id
 * shared across docs (the per-doc prefix already prevents true collisions, but
 * this is belt-and-suspenders) can never scroll to the wrong doc's heading.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  extractMarkdownHeadings,
  type MarkdownHeading,
} from "@/lib/markdown-headings";

export function MarkdownDocToc({
  body,
  docId,
  rootRef,
}: {
  body: string;
  docId?: string;
  /**
   * Ref to this doc's MarkdownDoc root element. Heading lookups are scoped to
   * this subtree so multiple docs on one page never cross-resolve.
   */
  rootRef: React.RefObject<HTMLElement | null>;
}) {
  const headings = useMemo(
    () => extractMarkdownHeadings(body, docId).filter((h) => h.depth >= 1 && h.depth <= 3),
    [body, docId],
  );
  const [activeId, setActiveId] = useState<string | null>(headings[0]?.id ?? null);
  const suppressIoUntilRef = useRef<number>(0);

  // Resolve a heading by id WITHIN this doc's subtree (never document-global).
  const findHeading = useCallback(
    (id: string): HTMLElement | null => {
      const root = rootRef.current;
      if (!root) return null;
      return root.querySelector<HTMLElement>(`[id="${cssEscapeId(id)}"]`);
    },
    [rootRef],
  );

  // Active-highlight on scroll (mirrors TaskTocSidebar).
  useEffect(() => {
    if (headings.length === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (Date.now() < suppressIoUntilRef.current) return;
        const viewportH = window.innerHeight;
        let best: { id: string; top: number } | null = null;
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const id = (e.target as HTMLElement).id;
          if (!id) continue;
          const top = e.boundingClientRect.top;
          if (top > viewportH * 0.6) continue;
          if (best === null || top < best.top) best = { id, top };
        }
        if (best) setActiveId(best.id);
      },
      {
        rootMargin: "0px 0px -40% 0px",
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
      },
    );
    // Headings may not exist yet on first paint (collapsible layer assigns
    // ids in a post-render effect); retry observation a couple of times.
    let raf = 0;
    let attempts = 0;
    const attach = () => {
      let found = 0;
      for (const h of headings) {
        const el = findHeading(h.id);
        if (el) {
          observer.observe(el);
          found++;
        }
      }
      if (found === 0 && attempts < 10) {
        attempts++;
        raf = window.requestAnimationFrame(attach);
      }
    };
    attach();
    return () => {
      observer.disconnect();
      if (raf) window.cancelAnimationFrame(raf);
    };
  }, [headings, findHeading]);

  const onClick = useCallback(
    (e: React.MouseEvent<HTMLAnchorElement>, h: MarkdownHeading) => {
      e.preventDefault();
      suppressIoUntilRef.current = Date.now() + 800;
      setActiveId(h.id);
      // Ask the collapsible layer to expand the target section (no-op if not
      // collapsible or already expanded). Scoped to this doc via docId.
      window.dispatchEvent(
        new CustomEvent("eps:section-expand", {
          detail: { headingId: h.id, docId },
        }),
      );
      // Scroll AFTER expand so the section's full height is committed.
      requestAnimationFrame(() => {
        const el = findHeading(h.id);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    },
    [docId, findHeading],
  );

  if (headings.length === 0) return null;

  // (cssEscapeId defined below the component)
  return (
    <nav
      aria-label="Table of contents"
      className="hidden @3xl:block sticky top-4 self-start max-h-[calc(100vh-2rem)] overflow-y-auto pr-2 text-xs"
    >
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
        Contents
      </div>
      <ul className="space-y-1">
        {headings.map((h) => {
          const isActive = h.id === activeId;
          return (
            <li key={`${h.id}-${h.index}`}>
              <a
                href={`#${h.id}`}
                onClick={(e) => onClick(e, h)}
                aria-current={isActive ? "true" : undefined}
                className={
                  "block rounded px-1.5 py-0.5 leading-snug " +
                  (isActive
                    ? "bg-stone-200 text-stone-900 font-medium"
                    : "text-stone-600 hover:bg-stone-100 hover:text-stone-900") +
                  " " +
                  (h.depth === 1
                    ? "font-semibold"
                    : h.depth === 2
                      ? "pl-3"
                      : "pl-6 text-stone-500")
                }
              >
                {h.text}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

/**
 * Escape an id for use inside a `[id="…"]` attribute selector. Our slugs are
 * `[a-z0-9-]` plus a `--` per-doc separator so this is defensive only, but
 * `CSS.escape` keeps a stray character from breaking the selector.
 */
function cssEscapeId(id: string): string {
  if (
    typeof window !== "undefined" &&
    (window as { CSS?: { escape?: typeof CSS.escape } }).CSS?.escape
  ) {
    return CSS.escape(id);
  }
  return id.replace(/["\\]/g, "\\$&");
}
