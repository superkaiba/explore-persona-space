import Link from "next/link";
import { listDocs } from "@/lib/docs";

export const dynamic = "force-dynamic";

export default function Docs() {
  const docs = listDocs();

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Docs</h1>
        <p className="mt-1 text-sm text-stone-600">
          Living research docs — open questions, literature, summaries. Source:{" "}
          <code className="rounded bg-stone-100 px-1">docs/*.md</code> in the repo.
        </p>
      </header>

      {docs.length === 0 ? (
        <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-6 text-center text-sm text-stone-500">
          No docs found in <code className="rounded bg-stone-100 px-1">docs/</code>.
        </p>
      ) : (
        <ul className="divide-y divide-stone-100 overflow-hidden rounded-lg border border-stone-200 bg-white">
          {docs.map((d) => (
            <li key={d.slug}>
              <Link
                href={`/docs/${d.slug}`}
                className="flex flex-col gap-1 px-4 py-3 hover:bg-stone-50 sm:px-5"
              >
                <span className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-stone-900">{d.title}</span>
                  {d.status && (
                    <span className="rounded bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700">
                      {d.status}
                    </span>
                  )}
                </span>
                {d.summary && (
                  <span className="text-sm leading-snug text-stone-600">{d.summary}</span>
                )}
                <span className="flex flex-wrap items-center gap-2 text-xs text-stone-500">
                  <span className="font-mono">{d.slug}.md</span>
                  {d.lastUpdated && <span>· updated {d.lastUpdated}</span>}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
