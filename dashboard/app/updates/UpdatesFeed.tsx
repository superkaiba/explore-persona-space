"use client";

/**
 * Client filter shell for the consolidated /updates pointer feed. Owns:
 *   - category chip state (Results / Mentor updates / Daily / Weekly)
 *   - date-range + search state, all synced to the URL so the view is
 *     shareable with the mentor
 *   - client-side filtering (category, date range, search)
 *
 * Each surviving item renders as a POINTER card: a single <Link> to the
 * item's canonical home (/results/<id> or /docs/<slug>). The feed never
 * re-renders the canonical body — that lives behind the link. No inline
 * comments, no lazy-loaded threads.
 *
 * Filter UX is adapted from the retired /log LogFeed, trimmed to the
 * pointer-card model.
 */
import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { UpdateFeedCategory, UpdateFeedItem } from "@/lib/logs";

type Chips = {
  cat: string | null; // raw comma-list e.g. "result,daily". null = all.
  from?: string; // ISO YYYY-MM-DD
  to?: string; // ISO YYYY-MM-DD
  q: string; // search term
};

const ALL_CAT_CHIPS: { key: UpdateFeedCategory; label: string }[] = [
  { key: "result", label: "Results" },
  { key: "mentor_updates", label: "Mentor updates" },
  { key: "daily", label: "Daily" },
  { key: "weekly", label: "Weekly" },
];

export function UpdatesFeed({
  items,
  initialChips,
}: {
  items: UpdateFeedItem[];
  initialChips: Chips;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Local mirror so typing in the search box doesn't round-trip per keystroke.
  const [searchDraft, setSearchDraft] = useState(initialChips.q);

  const activeCats: Set<UpdateFeedCategory> = useMemo(() => {
    const raw = searchParams.get("cat") ?? initialChips.cat;
    if (!raw) return new Set(ALL_CAT_CHIPS.map((c) => c.key));
    const wanted = new Set(raw.split(",").map((s) => s.trim().toLowerCase()));
    const out = new Set<UpdateFeedCategory>();
    for (const c of ALL_CAT_CHIPS) if (wanted.has(c.key)) out.add(c.key);
    // Empty set = nothing visible; treat as "all" so an unparseable URL can't
    // lock the user out of the feed.
    return out.size === 0 ? new Set(ALL_CAT_CHIPS.map((c) => c.key)) : out;
  }, [searchParams, initialChips.cat]);

  const from = searchParams.get("from") ?? initialChips.from;
  const to = searchParams.get("to") ?? initialChips.to;

  // Push a new URL state, preserving every param we don't override. Empty
  // strings get dropped so chip clears are clean.
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

  function toggleCat(cat: UpdateFeedCategory) {
    const allKeys = ALL_CAT_CHIPS.map((c) => c.key);
    const everythingOn = allKeys.every((k) => activeCats.has(k));
    // First click when "everything is on" means "only this category" — matches
    // the user's mental model of starting a fresh filter.
    let nextSet: Set<UpdateFeedCategory>;
    if (everythingOn) {
      nextSet = new Set([cat]);
    } else {
      nextSet = new Set(activeCats);
      if (nextSet.has(cat)) nextSet.delete(cat);
      else nextSet.add(cat);
      if (nextSet.size === 0) nextSet = new Set(allKeys);
    }
    const allOn = allKeys.every((k) => nextSet.has(k));
    updateParams({ cat: allOn ? null : Array.from(nextSet).join(",") });
  }

  function submitSearch() {
    updateParams({ q: searchDraft.trim() || null });
  }

  function clearAll() {
    setSearchDraft("");
    router.replace(pathname, { scroll: false });
  }

  const visible = useMemo(() => {
    const search = searchDraft.trim().toLowerCase();
    return items.filter((item) => {
      if (!activeCats.has(item.category)) return false;
      if (from && item.date < from) return false;
      if (to && item.date > to) return false;
      if (search) {
        const hay = `${item.title} ${item.excerpt ?? ""}`.toLowerCase();
        if (!hay.includes(search)) return false;
      }
      return true;
    });
  }, [items, activeCats, from, to, searchDraft]);

  const allCatsOn = ALL_CAT_CHIPS.every((c) => activeCats.has(c.key));

  return (
    <div className="space-y-4">
      {/* Chip row */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-stone-200 bg-white p-3">
        {ALL_CAT_CHIPS.map((c) => (
          <Chip
            key={c.key}
            active={activeCats.has(c.key) && !allCatsOn}
            onClick={() => toggleCat(c.key)}
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
            placeholder="Search title + excerpt"
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
            <li key={item.itemId}>
              <PointerCard item={item} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * Pointer card — a single link to the item's canonical home.
 * -------------------------------------------------------------------------- */

function PointerCard({ item }: { item: UpdateFeedItem }) {
  return (
    <Link
      href={item.href}
      className="block rounded-lg border border-stone-200 bg-white px-4 py-3 transition-colors hover:border-stone-300 hover:bg-stone-50 sm:px-5"
    >
      <div className="flex items-center gap-3">
        <CategoryBadge category={item.category} label={item.categoryLabel} />
        <time className="font-mono text-xs tabular-nums text-stone-500">{item.date}</time>
        <span className="flex-1 truncate text-sm font-medium leading-snug text-stone-900">
          {item.title}
        </span>
        {item.confidence && <ConfidenceBadge confidence={item.confidence} />}
        <span className="text-xs text-stone-400" aria-hidden="true">
          →
        </span>
      </div>
      {item.excerpt && (
        <p className="mt-1.5 line-clamp-2 text-sm leading-snug text-stone-600">{item.excerpt}</p>
      )}
    </Link>
  );
}

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  const activeCls = "border-stone-400 bg-stone-200 text-stone-900";
  const idleCls = "border-stone-300 bg-white text-stone-700 hover:bg-stone-50";
  return (
    <button
      type="button"
      onClick={onClick}
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

function CategoryBadge({
  category,
  label,
}: {
  category: UpdateFeedCategory;
  label: string;
}) {
  const palette: Record<UpdateFeedCategory, string> = {
    result: "bg-emerald-50 text-emerald-700",
    mentor_updates: "bg-amber-50 text-amber-700",
    daily: "bg-sky-50 text-sky-700",
    weekly: "bg-indigo-50 text-indigo-700",
  };
  return (
    <span className={`shrink-0 rounded px-2 py-0.5 text-xs font-medium ${palette[category]}`}>
      {label}
    </span>
  );
}

function ConfidenceBadge({
  confidence,
}: {
  confidence: "HIGH" | "MODERATE" | "LOW";
}) {
  const cls =
    confidence === "HIGH"
      ? "bg-emerald-50 text-emerald-700"
      : confidence === "MODERATE"
        ? "bg-amber-50 text-amber-700"
        : "bg-stone-100 text-stone-600";
  return (
    <span className={`shrink-0 rounded px-1.5 py-0.5 text-[11px] font-medium ${cls}`}>
      {confidence.toLowerCase()}
    </span>
  );
}
