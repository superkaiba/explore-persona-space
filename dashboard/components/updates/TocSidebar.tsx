"use client";

import { useCallback, useMemo } from "react";
import {
  extractMarkdownHeadings,
  type MarkdownHeading,
} from "@/lib/markdown-headings";

/**
 * Table-of-contents sidebar rendered next to the body in modal/fullscreen
 * views (`CardCommentBox` rail layout).
 *
 * - Lists H1/H2/H3 headings only. Deeper headings are excluded because
 *   they crowd the sidebar; if we ever need to surface H4 we add it here.
 * - Clicking an entry scrolls the matching `id="<slug>"` heading in the
 *   body into view. Heading ids are assigned by `MarkdownWithCollapsibleSections`
 *   (and by `renderMarkdownHeading` on the card-list path), using the same
 *   `githubLikeSlug` + `dedupeSlug` helpers, so anchors line up.
 * - On click, we also expand the section if it was collapsed, by toggling
 *   the localStorage key the collapsible layer reads — the layer listens
 *   for `eps:section-expand` CustomEvents.
 */
export function TocSidebar({ body, taskId }: { body: string; taskId: number }) {
  const headings = useMemo(
    () => extractMarkdownHeadings(body).filter((h) => h.depth >= 1 && h.depth <= 3),
    [body],
  );

  const onClick = useCallback(
    (e: React.MouseEvent<HTMLAnchorElement>, h: MarkdownHeading) => {
      e.preventDefault();
      const el = document.getElementById(h.id);
      if (!el) return;
      // Ask the collapsible layer to expand the target section first.
      // The CustomEvent is no-op if the section is already expanded.
      window.dispatchEvent(
        new CustomEvent("eps:section-expand", {
          detail: { headingId: h.id, taskId },
        }),
      );
      // Smooth scroll into view, centered.
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [taskId],
  );

  if (headings.length === 0) {
    return null;
  }

  return (
    <nav
      aria-label="Table of contents"
      className="hidden lg:block sticky top-0 self-start max-h-[calc(100vh-8rem)] overflow-y-auto pr-2 text-xs"
    >
      <div className="text-[10px] uppercase tracking-wider text-stone-500 mb-2 font-semibold">
        Contents
      </div>
      <ul className="space-y-1">
        {headings.map((h) => (
          <li key={`${h.id}-${h.index}`}>
            <a
              href={`#${h.id}`}
              onClick={(e) => onClick(e, h)}
              className={
                "block rounded px-1.5 py-0.5 text-stone-600 hover:bg-stone-100 hover:text-stone-900 leading-snug " +
                (h.depth === 1
                  ? "font-semibold"
                  : h.depth === 2
                    ? "pl-3 font-medium"
                    : "pl-6 text-stone-500")
              }
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
