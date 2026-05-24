import Link from "next/link";
import { listDailyBatches } from "@/lib/literature";

export const dynamic = "force-dynamic";

export default function Literature() {
  const batches = listDailyBatches();
  const totalItems = batches.reduce((n, b) => n + b.itemCount, 0);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Literature surfacing
        </h1>
        <p className="mt-1 text-sm text-stone-600">
          {batches.length} daily batch{batches.length === 1 ? "" : "es"} ·{" "}
          {totalItems} total surfaced items. Source: Sagan arXiv ranker, exported nightly.
        </p>
      </header>

      {batches.length === 0 ? (
        <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-6 text-center text-sm text-stone-500">
          No literature batches yet. Run{" "}
          <code className="rounded bg-stone-100 px-1">
            scripts/export_sagan_literature.py
          </code>
          .
        </p>
      ) : (
        <ul className="divide-y divide-stone-100 overflow-hidden rounded-lg border border-stone-200 bg-white">
          {batches.map((b) => (
            <li key={b.date}>
              <Link
                href={`/literature/${b.date}`}
                className="flex flex-col gap-1 px-4 py-3 hover:bg-stone-50 sm:flex-row sm:items-center sm:gap-4 sm:px-5"
              >
                <span className="font-mono text-sm text-stone-500 sm:w-32">
                  {b.date}
                </span>
                <span className="flex-1 text-sm leading-snug text-stone-900">
                  {b.itemCount} item{b.itemCount === 1 ? "" : "s"} surfaced
                </span>
                <span className="flex flex-wrap items-center gap-2">
                  <span className="rounded bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-700">
                    {b.itemCount} items
                  </span>
                  <span className="rounded bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700">
                    top {b.topScore}
                  </span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
