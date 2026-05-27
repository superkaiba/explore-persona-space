"use client";

/**
 * Left-rail TOC for `/tasks/[id]`. Renders one row per feed item in the
 * same reverse-chronological order as the main column.
 *
 * Click behavior:
 *   1. Dispatch `eps:feed-item-expand` so the matching CollapsiblePanel
 *      opens (no-op if already open).
 *   2. Smooth-scroll the panel's <section id=anchorId> into view.
 *
 * Active highlight: IntersectionObserver watches each panel's anchor
 * element. The topmost intersecting panel within the viewport's top
 * half wins the highlight, so as the user scrolls the bold row follows.
 *
 * Layout: sticky 240px-wide column, only rendered at md: and up via the
 * grid wrapper on the page. Hidden internally on narrow viewports as
 * defense-in-depth.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  FEED_ITEM_EXPAND_EVENT,
  type FeedItemExpandDetail,
} from "@/components/CollapsiblePanel";

export type TocEntry = {
  itemKey: string;
  anchorId: string;
  label: string;
  kind: "body" | "plan" | "event-card" | "event-compact" | "transition";
  ts: string;
};

const KIND_BADGE: Record<TocEntry["kind"], { label: string; cls: string }> = {
  body: { label: "BODY", cls: "bg-stone-200 text-stone-800" },
  plan: { label: "PLAN", cls: "bg-amber-100 text-amber-900" },
  "event-card": { label: "EVT", cls: "bg-sky-100 text-sky-800" },
  "event-compact": { label: "EVT", cls: "bg-stone-100 text-stone-600" },
  transition: { label: "→", cls: "bg-stone-100 text-stone-500" },
};

export function TaskTocSidebar({
  taskId,
  entries,
}: {
  taskId: number;
  entries: TocEntry[];
}) {
  const [activeKey, setActiveKey] = useState<string | null>(
    entries[0]?.itemKey ?? null,
  );
  // Avoid clobbering the active highlight while a programmatic scroll is in
  // flight: when the user clicks a TOC entry we set the active key
  // immediately and suppress IO updates for ~700ms (smooth-scroll duration).
  const suppressIoUntilRef = useRef<number>(0);

  useEffect(() => {
    if (entries.length === 0) return;
    const observer = new IntersectionObserver(
      (mutations) => {
        if (Date.now() < suppressIoUntilRef.current) return;
        // Each tick, snapshot the panel that's most-visibly the
        // "current section". We use the topmost panel whose top edge is
        // inside the viewport's upper half. This avoids the flicker
        // that pure ratio-based picking causes on tall sections.
        const viewportH = window.innerHeight;
        let best: { key: string; top: number } | null = null;
        for (const m of mutations) {
          if (!m.isIntersecting) continue;
          const el = m.target as HTMLElement;
          const key = el.dataset.feedItemKey;
          if (!key) continue;
          const top = m.boundingClientRect.top;
          if (top > viewportH * 0.6) continue;
          if (best === null || top < best.top) best = { key, top };
        }
        if (best) setActiveKey(best.key);
      },
      {
        rootMargin: "0px 0px -40% 0px",
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
      },
    );
    for (const e of entries) {
      const el = document.getElementById(e.anchorId);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [entries]);

  const onClick = useCallback(
    (e: React.MouseEvent<HTMLAnchorElement>, entry: TocEntry) => {
      e.preventDefault();
      suppressIoUntilRef.current = Date.now() + 800;
      setActiveKey(entry.itemKey);
      // Ask the matching panel to expand (no-op if already open).
      window.dispatchEvent(
        new CustomEvent<FeedItemExpandDetail>(FEED_ITEM_EXPAND_EVENT, {
          detail: { taskId, itemKey: entry.itemKey },
        }),
      );
      // Scroll AFTER expand so the panel's full height is committed; one
      // frame of delay is enough for React to flush the open state.
      requestAnimationFrame(() => {
        const el = document.getElementById(entry.anchorId);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    },
    [taskId],
  );

  if (entries.length === 0) return null;

  return (
    <nav
      aria-label="Task feed table of contents"
      className="hidden md:block sticky top-4 self-start max-h-[calc(100vh-2rem)] overflow-y-auto pr-2 text-xs"
    >
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
        Feed · {entries.length}
      </div>
      <ul className="space-y-0.5">
        {entries.map((entry) => {
          const isActive = entry.itemKey === activeKey;
          const badge = KIND_BADGE[entry.kind];
          return (
            <li key={entry.itemKey}>
              <a
                href={`#${entry.anchorId}`}
                onClick={(e) => onClick(e, entry)}
                aria-current={isActive ? "true" : undefined}
                className={
                  "flex items-baseline gap-2 rounded px-1.5 py-1 leading-snug " +
                  (isActive
                    ? "bg-stone-200 text-stone-900 font-medium"
                    : "text-stone-600 hover:bg-stone-100 hover:text-stone-900")
                }
              >
                <span
                  className={`shrink-0 rounded px-1 py-0.5 font-mono text-[9px] uppercase ${badge.cls}`}
                >
                  {badge.label}
                </span>
                <span className="min-w-0 flex-1 truncate">{entry.label}</span>
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
