"use client";

/**
 * Generic collapsible card shell.
 *
 * Drives the per-feed-row collapse UI on `/tasks/[id]`. Mirrors the
 * expand/collapse affordance used by `/log/LogCard` (chevron + click-the-
 * whole-header) but without the comment-thread / lazy-load coupling — this
 * component only handles "is the body visible".
 *
 * State persistence: keyed by `(taskId, itemKey)`, written to localStorage
 * under `eps:task-feed:<taskId>:<itemKey>`. The taskId/itemKey pair survives
 * page reloads AND avoids bleed between tasks. itemKey should be a stable
 * identifier (event ts+kind, "body", "plan-vN") — NOT a positional index,
 * which would shift when new events land.
 *
 * Cross-component coordination: listens for `eps:feed-item-expand`
 * CustomEvents (dispatched by `TaskTocSidebar` on click). When the
 * `(taskId, itemKey)` matches, force-expand and let the listener's caller
 * handle scrollIntoView on the wrapping <section id={anchorId}> element.
 *
 * Always-expanded mode: pass `alwaysExpanded` for items where collapse is
 * pure noise (transition pills are already 1-line). The header still
 * renders so the TOC anchor + click-to-scroll behavior works.
 */
import { ChevronDown, ChevronRight } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

export const FEED_COLLAPSE_STORAGE_PREFIX = "eps:task-feed:";
export const FEED_ITEM_EXPAND_EVENT = "eps:feed-item-expand";

export type FeedItemExpandDetail = {
  taskId: number;
  itemKey: string;
};

function readCollapseState(taskId: number, itemKey: string, fallback: boolean): boolean {
  try {
    const key = `${FEED_COLLAPSE_STORAGE_PREFIX}${taskId}:${itemKey}`;
    const v = window.localStorage.getItem(key);
    if (v === "1") return true;
    if (v === "0") return false;
    return fallback;
  } catch {
    return fallback;
  }
}

function writeCollapseState(taskId: number, itemKey: string, collapsed: boolean): void {
  try {
    const key = `${FEED_COLLAPSE_STORAGE_PREFIX}${taskId}:${itemKey}`;
    window.localStorage.setItem(key, collapsed ? "1" : "0");
  } catch {
    // localStorage unavailable / quota — silently ignore.
  }
}

export function CollapsiblePanel({
  taskId,
  itemKey,
  anchorId,
  header,
  children,
  defaultCollapsed = false,
  alwaysExpanded = false,
  emphasis,
}: {
  taskId: number;
  itemKey: string;
  anchorId: string;
  header: React.ReactNode;
  children: React.ReactNode;
  defaultCollapsed?: boolean;
  alwaysExpanded?: boolean;
  emphasis?: "plan";
}) {
  // SSR-safe: start with the SSR fallback (defaultCollapsed) so the server
  // markup matches the client's first render. Hydrate the persisted value
  // in a useEffect, which only runs client-side.
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  useEffect(() => {
    if (alwaysExpanded) return;
    // Hydration sync: SSR rendered with `defaultCollapsed`; on the
    // client we override with the persisted localStorage value (which
    // can't be read server-side). This is the canonical
    // "synchronize external state into React" effect — same pattern
    // `CommentableBody.tsx` uses for the section-collapse store.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setCollapsed(readCollapseState(taskId, itemKey, defaultCollapsed));
  }, [taskId, itemKey, defaultCollapsed, alwaysExpanded]);

  // Listen for TOC-driven expand requests. The TOC sidebar dispatches
  // `eps:feed-item-expand` with `{ taskId, itemKey }`; matching panels
  // open. Non-matching panels ignore the event.
  useEffect(() => {
    if (alwaysExpanded) return;
    const onExpand = (e: Event) => {
      const detail = (e as CustomEvent<FeedItemExpandDetail>).detail;
      if (!detail) return;
      if (detail.taskId !== taskId || detail.itemKey !== itemKey) return;
      setCollapsed(false);
      writeCollapseState(taskId, itemKey, false);
    };
    window.addEventListener(FEED_ITEM_EXPAND_EVENT, onExpand);
    return () => window.removeEventListener(FEED_ITEM_EXPAND_EVENT, onExpand);
  }, [taskId, itemKey, alwaysExpanded]);

  const onToggle = useCallback(() => {
    if (alwaysExpanded) return;
    setCollapsed((cur) => {
      const next = !cur;
      writeCollapseState(taskId, itemKey, next);
      return next;
    });
  }, [taskId, itemKey, alwaysExpanded]);

  // Plan cards get the same amber-tinted ring as the original page; other
  // cards get the neutral stone ring. Keep the existing visual register.
  const ring =
    emphasis === "plan"
      ? "border-stone-300 shadow-sm ring-1 ring-amber-100"
      : "border-stone-200";

  const isOpen = alwaysExpanded || !collapsed;

  return (
    <section
      id={anchorId}
      data-feed-item-key={itemKey}
      data-feed-item-collapsed={isOpen ? "false" : "true"}
      className={`scroll-mt-4 overflow-hidden rounded-lg border bg-white ${ring}`}
    >
      <button
        type="button"
        onClick={onToggle}
        disabled={alwaysExpanded}
        aria-expanded={isOpen}
        className={
          "flex w-full items-baseline gap-2 px-4 py-2 text-left text-xs text-stone-500 sm:px-6 " +
          (alwaysExpanded
            ? "cursor-default"
            : "cursor-pointer hover:bg-stone-50")
        }
      >
        {!alwaysExpanded && (
          <span className="text-stone-400" aria-hidden>
            {isOpen ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </span>
        )}
        <span className="flex flex-1 flex-wrap items-baseline gap-x-3 gap-y-1">
          {header}
        </span>
      </button>
      {isOpen && (
        <div className="border-t border-stone-100 px-4 pb-4 pt-3 sm:px-6 sm:pb-6">
          {children}
        </div>
      )}
    </section>
  );
}
