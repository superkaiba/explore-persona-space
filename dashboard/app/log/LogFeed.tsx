"use client";

/**
 * Client-side filter shell for the /log page. Owns:
 *   - chip state (kind multi-select, useful-only toggle, date range, search)
 *   - URL sync (chips encoded as query params so the view is shareable)
 *   - client-side filtering for "useful only" + search (date range is
 *     handled server-side because it changes which items are fetched)
 *
 * Hands each surviving item to <LogCard> for rendering + comments.
 */
import { useCallback, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { FeedItem, FeedItemKind } from "@/lib/logs";
import { LogCard } from "./LogCard";

type Chips = {
  kind: string | null;     // raw comma-list e.g. "daily,weekly". null = all.
  useful: boolean;         // true = hide classification:not-useful clean-results
  from?: string;           // ISO YYYY-MM-DD
  to?: string;             // ISO YYYY-MM-DD
  q: string;               // search term
};

const ALL_KIND_CHIPS: { key: FeedItemKind; label: string }[] = [
  { key: "daily", label: "Daily" },
  { key: "weekly", label: "Weekly" },
  { key: "ideation", label: "Ideation" },
  { key: "clean-result", label: "Results" },
];

export function LogFeed({
  items,
  initialChips,
  currentUserEmail,
}: {
  items: FeedItem[];
  initialChips: Chips;
  currentUserEmail: string | null;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Local mirror so typing in the search box doesn't pay for a full page
  // round-trip per keystroke. The kind chips + useful toggle + date
  // range all push back to URL on change (and re-fetch via router).
  const [searchDraft, setSearchDraft] = useState(initialChips.q);

  // Active kind set parsed from the URL (or initialChips on first paint).
  const activeKinds: Set<FeedItemKind> = useMemo(() => {
    const raw = searchParams.get("kind") ?? initialChips.kind;
    if (!raw) return new Set(ALL_KIND_CHIPS.map((c) => c.key));
    const wanted = new Set(
      raw
        .split(",")
        .map((s) => s.trim().toLowerCase()),
    );
    const out = new Set<FeedItemKind>();
    for (const c of ALL_KIND_CHIPS) if (wanted.has(c.key)) out.add(c.key);
    // Empty set = nothing visible; treat as "all" so the user can't lock
    // themselves out of the feed with an unparseable URL.
    return out.size === 0 ? new Set(ALL_KIND_CHIPS.map((c) => c.key)) : out;
  }, [searchParams, initialChips.kind]);

  const usefulOnly = useMemo(() => {
    const raw = searchParams.get("useful");
    if (raw === "all") return false;
    if (raw === null) return initialChips.useful;
    return true;
  }, [searchParams, initialChips.useful]);

  const from = searchParams.get("from") ?? initialChips.from;
  const to = searchParams.get("to") ?? initialChips.to;

  // Push a new URL state, preserving every param we don't explicitly
  // override. Empty strings get dropped so chip clears are clean.
  const updateParams = useCallback(
    (patch: Record<string, string | null>) => {
      const next = new URLSearchParams(searchParams.toString());
      for (const [k, v] of Object.entries(patch)) {
        if (v === null || v === "") next.delete(k);
        else next.set(k, v);
      }
      const qs = next.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  function toggleKind(kind: FeedItemKind) {
    const allKeys = ALL_KIND_CHIPS.map((c) => c.key);
    const everythingOn = allKeys.every((k) => activeKinds.has(k));
    // First click on a chip when "everything is on" should mean "only
    // this kind" — matches the user's mental model of a fresh filter.
    let nextSet: Set<FeedItemKind>;
    if (everythingOn) {
      nextSet = new Set([kind]);
    } else {
      nextSet = new Set(activeKinds);
      if (nextSet.has(kind)) nextSet.delete(kind);
      else nextSet.add(kind);
      // If the user clicks every chip off, fall back to "all" — same
      // safety net as activeKinds.
      if (nextSet.size === 0) nextSet = new Set(allKeys);
    }
    // Encode as "all" → no param. Otherwise comma-list.
    const allOn = allKeys.every((k) => nextSet.has(k));
    updateParams({ kind: allOn ? null : Array.from(nextSet).join(",") });
  }

  function toggleUseful() {
    // useful-only is the default; the URL param only appears when we
    // *want* to show not-useful too.
    updateParams({ useful: usefulOnly ? "all" : null });
  }

  function submitSearch() {
    updateParams({ q: searchDraft.trim() || null });
  }

  function clearAll() {
    setSearchDraft("");
    router.replace(pathname, { scroll: false });
  }

  // Apply the client-side filters: kind, useful-only, search.
  const visible = useMemo(() => {
    const search = searchDraft.trim().toLowerCase();
    return items.filter((item) => {
      if (!activeKinds.has(item.kind as FeedItemKind)) return false;
      if (
        usefulOnly &&
        item.kind === "clean-result" &&
        item.classification === "not-useful"
      ) {
        return false;
      }
      if (search) {
        const hay = `${item.title} ${item.kind === "clean-result" ? "" : (item as { tags?: string[] }).tags?.join(" ") ?? ""} ${item.body}`.toLowerCase();
        if (!hay.includes(search)) return false;
      }
      return true;
    });
  }, [items, activeKinds, usefulOnly, searchDraft]);

  const allKindsOn = ALL_KIND_CHIPS.every((c) => activeKinds.has(c.key));

  return (
    <div className="space-y-4">
      {/* Chip row */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-stone-200 bg-white p-3">
        <Chip
          active={usefulOnly}
          onClick={toggleUseful}
          tone="emerald"
          title={
            usefulOnly
              ? "Showing only useful clean-results. Click to include not-useful too."
              : "Showing all clean-results. Click to filter to useful only."
          }
        >
          {usefulOnly ? "Useful only ✓" : "Useful only"}
        </Chip>

        <div className="mx-1 h-5 w-px bg-stone-200" />

        {ALL_KIND_CHIPS.map((c) => (
          <Chip
            key={c.key}
            active={activeKinds.has(c.key) && !allKindsOn}
            onClick={() => toggleKind(c.key)}
            tone="stone"
          >
            {c.label}
          </Chip>
        ))}

        <div className="mx-1 h-5 w-px bg-stone-200" />

        <DateInput
          label="From"
          value={from ?? ""}
          onChange={(v) => updateParams({ from: v || null })}
        />
        <DateInput
          label="To"
          value={to ?? ""}
          onChange={(v) => updateParams({ to: v || null })}
        />

        <div className="mx-1 h-5 w-px bg-stone-200" />

        <div className="flex items-center gap-1">
          <input
            type="search"
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submitSearch();
              }
            }}
            placeholder="Search title + body"
            className="w-44 rounded border border-stone-300 bg-white px-2 py-1 text-sm"
          />
          {searchDraft && (
            <button
              type="button"
              onClick={() => {
                setSearchDraft("");
                updateParams({ q: null });
              }}
              className="rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-700"
              aria-label="Clear search"
              title="Clear search"
            >
              ×
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={clearAll}
          className="ml-auto rounded px-2 py-1 text-xs text-stone-500 hover:bg-stone-100 hover:text-stone-800"
        >
          Reset all
        </button>
      </div>

      {/* Count */}
      <div className="text-xs text-stone-500">
        {visible.length} item{visible.length === 1 ? "" : "s"}
        {visible.length !== items.length && (
          <span className="text-stone-400"> (of {items.length})</span>
        )}
      </div>

      {/* Feed */}
      {visible.length === 0 ? (
        <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-8 text-center text-sm text-stone-500">
          No entries match the current filters.
        </p>
      ) : (
        <ul className="space-y-3">
          {visible.map((item) => (
            <li key={item.entryId}>
              <LogCard item={item} currentUserEmail={currentUserEmail} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Chip({
  active,
  onClick,
  tone,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  tone: "emerald" | "stone";
  title?: string;
  children: React.ReactNode;
}) {
  const activeCls =
    tone === "emerald"
      ? "border-emerald-300 bg-emerald-100 text-emerald-900"
      : "border-stone-400 bg-stone-200 text-stone-900";
  const idleCls = "border-stone-300 bg-white text-stone-700 hover:bg-stone-50";
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`rounded-full border px-2.5 py-1 text-xs font-medium transition-colors ${
        active ? activeCls : idleCls
      }`}
    >
      {children}
    </button>
  );
}

function DateInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1 text-xs text-stone-500">
      <span>{label}</span>
      <input
        type="date"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-stone-300 bg-white px-1.5 py-0.5 text-xs"
      />
    </label>
  );
}
