"use client";

/**
 * Client filter shell for /questions.
 *
 * Server (page.tsx) renders the full grouped list (all questions + apps) into
 * the initial HTML so the deep-link `#q-<slug>` anchor resolves on first
 * paint EVEN WHEN the matching question would be hidden by the default
 * filter. The client then reads `?show=all|open` (defaulting "all" when the
 * URL carries a `#q-...` hash, "open" otherwise) and toggles visibility
 * per-row via CSS — never unmounts the row, so the browser's automatic
 * hash-scroll still lands.
 *
 * The filter mode is DERIVED FROM THE URL via `useSyncExternalStore`: the
 * server snapshot is always "all" (so SSR HTML == first client paint — no
 * hydration mismatch), and the client snapshot reads the URL. React applies
 * the URL-derived mode in a post-hydration re-render, which is exactly what
 * `useSyncExternalStore`'s server/client snapshot split is for (and avoids
 * both the hydration warning AND the set-state-in-effect lint). Flipping the
 * toggle writes `?show=` and notifies the store; a `useEffect` keyed on the
 * resulting mode does the CSS show/hide (DOM work, not state).
 */
import { useEffect, useSyncExternalStore } from "react";

type Mode = "open" | "all";

const MODE_CHANGE_EVENT = "questions:modechange";

/** Client snapshot: the mode the URL currently implies. */
function modeFromUrl(): Mode {
  if (typeof window === "undefined") return "all";
  const url = new URL(window.location.href);
  const raw = url.searchParams.get("show");
  if (raw === "open" || raw === "all") return raw;
  // Default to "all" when a `#q-...` deep-link is present so the target
  // question is never filtered out on land. Otherwise default to "open".
  if (window.location.hash.startsWith("#q-")) return "all";
  return "open";
}

/** Server snapshot: deterministic, matches first paint (full list visible). */
function serverMode(): Mode {
  return "all";
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener("popstate", onChange);
  window.addEventListener("hashchange", onChange);
  window.addEventListener(MODE_CHANGE_EVENT, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener("hashchange", onChange);
    window.removeEventListener(MODE_CHANGE_EVENT, onChange);
  };
}

export function QuestionsBrowser() {
  const mode = useSyncExternalStore(subscribe, modeFromUrl, serverMode);

  // Toggle handler: write the choice to the URL (so it survives reloads + is
  // shareable) and notify the store, which re-reads `modeFromUrl()`. No
  // setState — the URL is the source of truth.
  function chooseMode(next: Mode) {
    const url = new URL(window.location.href);
    url.searchParams.set("show", next);
    window.history.replaceState(null, "", url.toString());
    window.dispatchEvent(new Event(MODE_CHANGE_EVENT));
  }

  useEffect(() => {
    // Apply visibility to every question + group container, then suppress
    // groups that end up empty under the current filter. CSS does the
    // hide/show so the row stays in the DOM and #q-<slug> targeting works.
    if (typeof document === "undefined") return;
    const rows = document.querySelectorAll<HTMLElement>("[data-q-status]");
    rows.forEach((row) => {
      const status = row.dataset.qStatus;
      const kind = row.dataset.qKind;
      // Applications are always shown — they're not status-filterable.
      const visible =
        mode === "all" || kind === "application" || status === "open";
      row.style.display = visible ? "" : "none";
    });
    const groups = document.querySelectorAll<HTMLElement>("[data-q-group]");
    groups.forEach((group) => {
      const groupRows =
        group.querySelectorAll<HTMLElement>("[data-q-status]");
      let visibleCount = 0;
      groupRows.forEach((row) => {
        if (row.style.display !== "none") visibleCount++;
      });
      group.style.display = visibleCount === 0 ? "none" : "";
    });

    // After flipping the filter, re-scroll to the hash if it still maps to a
    // visible element (the browser's initial scroll happened before our
    // visibility pass).
    if (window.location.hash.startsWith("#q-")) {
      const target = document.getElementById(window.location.hash.slice(1));
      if (target && target.offsetParent !== null) {
        target.scrollIntoView({ behavior: "instant", block: "start" });
      }
    }
  }, [mode]);

  return (
    <div
      role="group"
      aria-label="Question status filter"
      className="inline-flex items-center rounded-md border border-stone-200 bg-white p-0.5 text-xs"
    >
      <button
        type="button"
        onClick={() => chooseMode("open")}
        className={
          "rounded px-2.5 py-1 font-medium transition-colors " +
          (mode === "open"
            ? "bg-stone-900 text-white"
            : "text-stone-600 hover:text-stone-900")
        }
        aria-pressed={mode === "open"}
      >
        Open only
      </button>
      <button
        type="button"
        onClick={() => chooseMode("all")}
        className={
          "rounded px-2.5 py-1 font-medium transition-colors " +
          (mode === "all"
            ? "bg-stone-900 text-white"
            : "text-stone-600 hover:text-stone-900")
        }
        aria-pressed={mode === "all"}
      >
        All
      </button>
    </div>
  );
}
