"use client";

/**
 * Client-side filter + browse shell for the public Results catalog.
 *
 * Pure client filtering over a pre-loaded, public-safe `ResultListing[]`
 * (the server already applied the authoritative completed+`useful` predicate,
 * so there is nothing private to leak here). Filters: confidence
 * (HIGH/MODERATE/LOW), tag/topic, and a coarse date window. Free-text search
 * narrows on title + excerpt.
 */
import { useMemo, useState } from "react";
import Link from "next/link";
import type { ResultConfidence, ResultListing } from "@/lib/results";

const CONFIDENCE_ORDER: Exclude<ResultConfidence, null>[] = [
  "HIGH",
  "MODERATE",
  "LOW",
];

const CONFIDENCE_STYLE: Record<
  Exclude<ResultConfidence, null>,
  { chip: string; dot: string }
> = {
  HIGH: { chip: "bg-emerald-50 text-emerald-700 border-emerald-200", dot: "bg-emerald-500" },
  MODERATE: { chip: "bg-amber-50 text-amber-800 border-amber-200", dot: "bg-amber-500" },
  LOW: { chip: "bg-stone-100 text-stone-600 border-stone-200", dot: "bg-stone-400" },
};

type DateWindow = "all" | "7" | "30" | "90";

const DATE_WINDOWS: { value: DateWindow; label: string }[] = [
  { value: "all", label: "All time" },
  { value: "7", label: "Last 7 days" },
  { value: "30", label: "Last 30 days" },
  { value: "90", label: "Last 90 days" },
];

function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function ResultsBrowser({
  results,
  allTags,
}: {
  results: ResultListing[];
  allTags: string[];
}) {
  const [query, setQuery] = useState("");
  const [confidence, setConfidence] = useState<ResultConfidence>(null);
  const [tag, setTag] = useState<string | null>(null);
  const [dateWindow, setDateWindow] = useState<DateWindow>("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const cutoff =
      dateWindow === "all"
        ? null
        : Date.now() - Number(dateWindow) * 24 * 60 * 60 * 1000;
    return results.filter((r) => {
      if (confidence && r.confidence !== confidence) return false;
      if (tag && !r.tags.includes(tag)) return false;
      if (cutoff != null) {
        const t = Date.parse(r.date);
        if (Number.isFinite(t) && t < cutoff) return false;
      }
      if (q) {
        const hay = `${r.rawTitle}\n${r.excerpt}\n${r.tags.join(" ")}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [results, query, confidence, tag, dateWindow]);

  const confidenceCounts = useMemo(() => {
    const counts: Record<string, number> = { HIGH: 0, MODERATE: 0, LOW: 0 };
    for (const r of results) {
      if (r.confidence) counts[r.confidence] = (counts[r.confidence] ?? 0) + 1;
    }
    return counts;
  }, [results]);

  const anyFilter = !!query || !!confidence || !!tag || dateWindow !== "all";

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="space-y-3 rounded-lg border border-stone-200 bg-white p-4">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search results by title, finding, or tag…"
          className="w-full rounded-md border border-stone-300 px-3 py-2 text-sm focus:border-stone-500 focus:outline-none focus:ring-1 focus:ring-stone-400"
          aria-label="Search results"
        />

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium uppercase tracking-wide text-stone-400">
            Confidence
          </span>
          <button
            type="button"
            onClick={() => setConfidence(null)}
            className={chipClass(confidence === null)}
          >
            Any
          </button>
          {CONFIDENCE_ORDER.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setConfidence(confidence === c ? null : c)}
              className={chipClass(confidence === c)}
            >
              <span
                className={`mr-1.5 inline-block h-2 w-2 rounded-full ${CONFIDENCE_STYLE[c].dot}`}
              />
              {c.charAt(0) + c.slice(1).toLowerCase()}
              <span className="ml-1 text-stone-400">{confidenceCounts[c] ?? 0}</span>
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium uppercase tracking-wide text-stone-400">
            Date
          </span>
          {DATE_WINDOWS.map((w) => (
            <button
              key={w.value}
              type="button"
              onClick={() => setDateWindow(w.value)}
              className={chipClass(dateWindow === w.value)}
            >
              {w.label}
            </button>
          ))}
        </div>

        {allTags.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-wide text-stone-400">
              Topic
            </span>
            <button
              type="button"
              onClick={() => setTag(null)}
              className={chipClass(tag === null)}
            >
              All
            </button>
            {allTags.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTag(tag === t ? null : t)}
                className={chipClass(tag === t)}
              >
                #{t}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Result count + clear */}
      <div className="flex items-center justify-between text-sm text-stone-500">
        <span>
          {filtered.length} result{filtered.length === 1 ? "" : "s"}
          {anyFilter ? ` of ${results.length}` : ""}
        </span>
        {anyFilter && (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setConfidence(null);
              setTag(null);
              setDateWindow("all");
            }}
            className="text-stone-500 underline-offset-2 hover:text-stone-800 hover:underline"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Cards */}
      {filtered.length === 0 ? (
        <p className="rounded-lg border border-dashed border-stone-300 bg-white px-4 py-10 text-center text-sm text-stone-500">
          {results.length === 0
            ? "No published results yet."
            : "No results match these filters."}
        </p>
      ) : (
        <ul className="grid gap-4 sm:grid-cols-2">
          {filtered.map((r) => (
            <li key={r.id}>
              <ResultCard result={r} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ResultCard({ result }: { result: ResultListing }) {
  const date = formatDate(result.date);
  return (
    <Link
      href={result.href}
      className="flex h-full flex-col gap-3 rounded-lg border border-stone-200 bg-white p-5 transition-colors hover:border-stone-300 hover:bg-stone-50"
    >
      <div className="flex items-center gap-2 text-xs text-stone-500">
        <span className="font-mono">#{result.id}</span>
        {date && (
          <>
            <span aria-hidden>·</span>
            <time>{date}</time>
          </>
        )}
        {result.confidence && (
          <span className="ml-auto">
            <ConfidenceBadge confidence={result.confidence} />
          </span>
        )}
      </div>

      <h2 className="text-base font-semibold leading-snug tracking-tight text-stone-900">
        {result.title}
      </h2>

      {result.excerpt && (
        <p className="line-clamp-3 text-sm leading-relaxed text-stone-600">
          {result.excerpt}
        </p>
      )}

      {result.tags.length > 0 && (
        <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
          {result.tags.map((t) => (
            <span
              key={t}
              className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600"
            >
              #{t}
            </span>
          ))}
        </div>
      )}
    </Link>
  );
}

function ConfidenceBadge({
  confidence,
}: {
  confidence: Exclude<ResultConfidence, null>;
}) {
  const style = CONFIDENCE_STYLE[confidence];
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium ${style.chip}`}
    >
      <span className={`mr-1.5 inline-block h-2 w-2 rounded-full ${style.dot}`} />
      {confidence.charAt(0) + confidence.slice(1).toLowerCase()} confidence
    </span>
  );
}

function chipClass(active: boolean): string {
  return [
    "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
    active
      ? "border-stone-800 bg-stone-900 text-white"
      : "border-stone-200 bg-white text-stone-600 hover:border-stone-300 hover:bg-stone-50",
  ].join(" ");
}
