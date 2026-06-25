"use client";

/**
 * <DataTable> — the sort / filter / search / reveal-more table for ONE data
 * artifact (a figure's resolved row set), part of the interactive data viewer
 * (clean-result v4 redesign, Phase 2).
 *
 * Pure presentational + client-interactive: it receives an already-resolved
 * `DataArtifact` (rows + column schema, fetched by the parent <TaskDataViewer>
 * from GET /tasks/<id>/data) and provides:
 *
 *   - SORT: click a column header to sort by it (numeric columns sort
 *     numerically, others lexically); click again to flip direction, a third
 *     time to clear. The active column shows an up/down arrow.
 *   - FILTER: a per-column value picker appears for low-cardinality columns
 *     (≤ FILTER_MAX_DISTINCT distinct values) — pick one or more values to
 *     keep. Combined across columns with AND.
 *   - SEARCH: a free-text box matches the substring (case-insensitive) against
 *     any cell in the row.
 *   - REVEAL-MORE: rows render in pages of PAGE_SIZE; "Show more" reveals the
 *     next page, "Show all" reveals everything fetched. The body's v4 sample is
 *     a tiny subset; this is the "load more" the Phase-2 contract calls for,
 *     over the rows that ARE on local disk.
 *
 * Whatever is NOT on local disk (the full set behind external HF links) is the
 * parent's link-out responsibility; this table never fabricates rows.
 */
import { useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  ListFilter,
  Search,
  X,
} from "lucide-react";
import type { DataColumn } from "@/lib/task-data";

const PAGE_SIZE = 25;
// A column gets a value-filter dropdown only when it has at most this many
// distinct values — otherwise the dropdown is a wall of unique numbers.
const FILTER_MAX_DISTINCT = 30;

type Row = Record<string, unknown>;
type SortDir = "asc" | "desc";

export function DataTable({
  columns,
  rows,
}: {
  columns: DataColumn[];
  rows: Row[];
}) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [search, setSearch] = useState("");
  // Per-column value filter: column key -> set of allowed string-rendered cells.
  const [filters, setFilters] = useState<Record<string, Set<string>>>({});
  const [openFilter, setOpenFilter] = useState<string | null>(null);
  const [visible, setVisible] = useState(PAGE_SIZE);

  const colType = useMemo(() => {
    const m = new Map<string, DataColumn["type"]>();
    for (const c of columns) m.set(c.key, c.type);
    return m;
  }, [columns]);

  // Distinct values per column (only computed for columns small enough to
  // offer a dropdown; memoized once per rows/columns change).
  const distinctByCol = useMemo(() => {
    const out = new Map<string, string[]>();
    for (const c of columns) {
      const seen = new Set<string>();
      for (const r of rows) {
        seen.add(cellStr(r[c.key]));
        if (seen.size > FILTER_MAX_DISTINCT) break;
      }
      if (seen.size <= FILTER_MAX_DISTINCT) {
        out.set(c.key, Array.from(seen).sort(compareStrings));
      }
    }
    return out;
  }, [columns, rows]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const activeFilters = Object.entries(filters).filter(([, s]) => s.size > 0);
    return rows.filter((r) => {
      if (needle) {
        const hit = columns.some((c) => cellStr(r[c.key]).toLowerCase().includes(needle));
        if (!hit) return false;
      }
      for (const [key, allowed] of activeFilters) {
        if (!allowed.has(cellStr(r[key]))) return false;
      }
      return true;
    });
  }, [rows, columns, search, filters]);

  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    const numeric = colType.get(sortKey) === "number";
    const dir = sortDir === "asc" ? 1 : -1;
    // Copy before sort — never mutate the filtered array in place.
    return [...filtered].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      // Empty cells always sort last regardless of direction.
      const aEmpty = av === null || av === undefined || av === "";
      const bEmpty = bv === null || bv === undefined || bv === "";
      if (aEmpty && bEmpty) return 0;
      if (aEmpty) return 1;
      if (bEmpty) return -1;
      if (numeric) {
        return (Number(av) - Number(bv)) * dir;
      }
      return compareStrings(cellStr(av), cellStr(bv)) * dir;
    });
  }, [filtered, sortKey, sortDir, colType]);

  const shown = sorted.slice(0, visible);
  const activeFilterCount = Object.values(filters).reduce(
    (n, s) => n + (s.size > 0 ? 1 : 0),
    0,
  );

  function onHeaderClick(key: string) {
    setVisible(PAGE_SIZE);
    if (sortKey !== key) {
      setSortKey(key);
      setSortDir("asc");
      return;
    }
    if (sortDir === "asc") {
      setSortDir("desc");
      return;
    }
    // Third click clears the sort.
    setSortKey(null);
    setSortDir("asc");
  }

  function toggleFilterValue(key: string, value: string) {
    setVisible(PAGE_SIZE);
    setFilters((prev) => {
      const next = { ...prev };
      const set = new Set(next[key] ?? []);
      if (set.has(value)) set.delete(value);
      else set.add(value);
      if (set.size === 0) delete next[key];
      else next[key] = set;
      return next;
    });
  }

  function clearAll() {
    setSearch("");
    setFilters({});
    setSortKey(null);
    setSortDir("asc");
    setVisible(PAGE_SIZE);
    setOpenFilter(null);
  }

  const hasControls = search.trim() !== "" || activeFilterCount > 0 || sortKey !== null;

  return (
    <div className="space-y-3">
      {/* Controls row: search + active-filter chips + clear. */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-stone-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setVisible(PAGE_SIZE);
            }}
            placeholder="Search all columns…"
            className="w-56 rounded border border-stone-300 bg-white py-1 pl-7 pr-2 text-xs text-stone-800 placeholder:text-stone-400 focus:border-stone-500 focus:outline-none"
          />
        </div>
        {activeFilterCount > 0 && (
          <span className="inline-flex items-center gap-1 rounded bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-900">
            <ListFilter className="h-3 w-3" />
            {activeFilterCount} filter{activeFilterCount === 1 ? "" : "s"}
          </span>
        )}
        {hasControls && (
          <button
            type="button"
            onClick={clearAll}
            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-stone-500 hover:bg-stone-100 hover:text-stone-800"
          >
            <X className="h-3 w-3" />
            Reset
          </button>
        )}
        <span className="ml-auto text-[11px] tabular-nums text-stone-500">
          {filtered.length === rows.length
            ? `${rows.length} rows`
            : `${filtered.length} of ${rows.length} rows`}
        </span>
      </div>

      {/* The table. Horizontal scroll for wide schemas. */}
      <div className="overflow-x-auto rounded border border-stone-200">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr className="border-b border-stone-200 bg-stone-50">
              {columns.map((c) => {
                const active = sortKey === c.key;
                const distinct = distinctByCol.get(c.key);
                const filterable = distinct !== undefined && distinct.length > 1;
                const filterActive = (filters[c.key]?.size ?? 0) > 0;
                return (
                  <th
                    key={c.key}
                    className="relative whitespace-nowrap px-2.5 py-1.5 text-left font-semibold text-stone-700"
                  >
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => onHeaderClick(c.key)}
                        className={
                          "inline-flex items-center gap-1 hover:text-stone-950 " +
                          (c.type === "number" ? "tabular-nums" : "")
                        }
                        title={`Sort by ${c.key}`}
                      >
                        <span>{c.key}</span>
                        {active ? (
                          sortDir === "asc" ? (
                            <ArrowUp className="h-3 w-3 text-stone-600" />
                          ) : (
                            <ArrowDown className="h-3 w-3 text-stone-600" />
                          )
                        ) : (
                          <ArrowUpDown className="h-3 w-3 text-stone-300" />
                        )}
                      </button>
                      {filterable && (
                        <button
                          type="button"
                          onClick={() =>
                            setOpenFilter((cur) => (cur === c.key ? null : c.key))
                          }
                          className={
                            "rounded p-0.5 hover:bg-stone-200 " +
                            (filterActive ? "text-amber-700" : "text-stone-300")
                          }
                          title={`Filter ${c.key}`}
                          aria-label={`Filter ${c.key}`}
                        >
                          <ListFilter className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                    {filterable && openFilter === c.key && (
                      <FilterDropdown
                        values={distinct}
                        selected={filters[c.key] ?? new Set()}
                        onToggle={(v) => toggleFilterValue(c.key, v)}
                        onClose={() => setOpenFilter(null)}
                      />
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr
                key={i}
                className="border-b border-stone-100 last:border-0 hover:bg-amber-50/40"
              >
                {columns.map((c) => {
                  const v = r[c.key];
                  return (
                    <td
                      key={c.key}
                      className={
                        "max-w-[28rem] truncate px-2.5 py-1 align-top text-stone-700 " +
                        (c.type === "number" ? "tabular-nums" : "")
                      }
                      title={cellStr(v)}
                    >
                      {cellStr(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
            {shown.length === 0 && (
              <tr>
                <td
                  colSpan={Math.max(1, columns.length)}
                  className="px-3 py-6 text-center text-stone-500"
                >
                  No rows match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Reveal-more controls. */}
      {sorted.length > visible && (
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setVisible((v) => v + PAGE_SIZE)}
            className="rounded border border-stone-300 bg-white px-3 py-1 text-xs font-medium text-stone-700 hover:bg-stone-50"
          >
            Show {Math.min(PAGE_SIZE, sorted.length - visible)} more
          </button>
          <button
            type="button"
            onClick={() => setVisible(sorted.length)}
            className="rounded px-2 py-1 text-xs text-stone-500 hover:bg-stone-100 hover:text-stone-800"
          >
            Show all {sorted.length}
          </button>
        </div>
      )}
    </div>
  );
}

function FilterDropdown({
  values,
  selected,
  onToggle,
  onClose,
}: {
  values: string[];
  selected: Set<string>;
  onToggle: (v: string) => void;
  onClose: () => void;
}) {
  return (
    <>
      {/* Click-away backdrop. */}
      <div className="fixed inset-0 z-10" onClick={onClose} aria-hidden />
      <div className="absolute left-0 top-full z-20 mt-1 max-h-64 w-56 overflow-auto rounded-md border border-stone-300 bg-white p-1 shadow-lg">
        {values.map((v) => {
          const checked = selected.has(v);
          return (
            <label
              key={v}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-xs font-normal text-stone-700 hover:bg-stone-100"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggle(v)}
                className="h-3 w-3 accent-amber-600"
              />
              <span className="truncate" title={v}>
                {v === "" ? <em className="text-stone-400">(empty)</em> : v}
              </span>
            </label>
          );
        })}
      </div>
    </>
  );
}

// ── helpers ───────────────────────────────────────────────────────────────────

/** Render a cell value as a display string (null/undefined -> ""). */
function cellStr(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") {
    // Trim float noise without losing precision on integers / short decimals.
    if (Number.isInteger(v)) return String(v);
    return String(Number(v.toPrecision(6)));
  }
  return String(v);
}

/** Natural-ish string compare: numeric-aware, case-insensitive. */
function compareStrings(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}
