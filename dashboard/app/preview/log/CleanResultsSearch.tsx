"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

type Row = {
  taskId: number;
  title: string;
  date: string;
  classification: "useful" | "not-useful" | "pending";
  body: string;
};

export function CleanResultsSearch({ rows }: { rows: Row[] }) {
  const [q, setQ] = useState("");
  const [includeNotUseful, setIncludeNotUseful] = useState(true);
  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return rows.filter((r) => {
      if (!includeNotUseful && r.classification === "not-useful") return false;
      if (!needle) return true;
      const hay = `${r.title}\n${r.body}`.toLowerCase();
      return hay.includes(needle);
    });
  }, [rows, q, includeNotUseful]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search title and body..."
          className="flex-1 rounded-lg border border-stone-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-stone-500 focus:outline-none"
        />
        <label className="flex items-center gap-2 text-sm text-stone-600">
          <input
            type="checkbox"
            checked={includeNotUseful}
            onChange={(e) => setIncludeNotUseful(e.target.checked)}
            className="h-4 w-4 rounded border-stone-300"
          />
          include &quot;not useful&quot;
        </label>
      </div>

      <div className="text-xs text-stone-500">
        {filtered.length} of {rows.length} shown
      </div>

      <ul className="space-y-2">
        {filtered.map((r) => (
          <li
            key={r.taskId}
            className="rounded-lg border border-stone-200 bg-white"
          >
            <Link
              href={`/tasks/${r.taskId}`}
              className="flex flex-col gap-2 px-4 py-3 hover:bg-stone-50 sm:flex-row sm:items-start sm:gap-4 sm:px-5"
            >
              <span className="font-mono text-sm text-stone-500 sm:w-24">
                #{r.taskId}
              </span>
              <span className="flex-1 text-sm leading-snug text-stone-900">
                {r.title}
                <Highlight body={r.body} q={q} />
              </span>
              <span className="flex items-center gap-2 whitespace-nowrap">
                <ClassificationBadge c={r.classification} />
                <span className="font-mono text-xs text-stone-500">{r.date}</span>
              </span>
            </Link>
          </li>
        ))}
      </ul>

      {filtered.length === 0 && (
        <div className="rounded-lg border border-dashed border-stone-200 bg-white px-5 py-8 text-center text-sm italic text-stone-400">
          no clean results match
        </div>
      )}
    </div>
  );
}

function ClassificationBadge({
  c,
}: {
  c: "useful" | "not-useful" | "pending";
}) {
  const cls =
    c === "useful"
      ? "bg-emerald-50 text-emerald-700"
      : c === "not-useful"
        ? "bg-rose-50 text-rose-700"
        : "bg-stone-100 text-stone-600";
  const label = c === "useful" ? "useful" : c === "not-useful" ? "not useful" : "pending";
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}

function Highlight({ body, q }: { body: string; q: string }) {
  const needle = q.trim().toLowerCase();
  if (!needle) return null;
  const hay = body.toLowerCase();
  const idx = hay.indexOf(needle);
  if (idx < 0) return null;
  const start = Math.max(0, idx - 60);
  const end = Math.min(body.length, idx + needle.length + 60);
  const before = body.slice(start, idx);
  const hit = body.slice(idx, idx + needle.length);
  const after = body.slice(idx + needle.length, end);
  return (
    <div className="mt-1 text-xs text-stone-500">
      {start > 0 && "…"}
      {before}
      <mark className="rounded bg-amber-100 px-0.5 text-stone-900">{hit}</mark>
      {after}
      {end < body.length && "…"}
    </div>
  );
}
