import Link from "next/link";
import { listDocsByCategory } from "@/lib/docs";

export const dynamic = "force-dynamic";

export default function Docs() {
  const groups = listDocsByCategory();
  const total = groups.reduce((n, g) => n + g.docs.length, 0);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Docs</h1>
        <p className="mt-1 text-sm text-stone-600">
          Living research docs, meeting + mentor updates, activity logs, and ideas. Sources:{" "}
          <code className="rounded bg-stone-100 px-1">docs/*.md</code>,{" "}
          <code className="rounded bg-stone-100 px-1">docs/mentor_updates/</code>,{" "}
          <code className="rounded bg-stone-100 px-1">logs/daily</code> +{" "}
          <code className="rounded bg-stone-100 px-1">logs/weekly</code>,{" "}
          <code className="rounded bg-stone-100 px-1">docs/ideas/</code>.
        </p>
      </header>

      {total === 0 ? (
        <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-6 text-center text-sm text-stone-500">
          No docs found.
        </p>
      ) : (
        <div className="space-y-8">
          {groups.map((group) => (
            <section key={group.category} className="space-y-3">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                {group.category}{" "}
                <span className="font-normal text-stone-400">({group.docs.length})</span>
              </h2>
              <ul className="divide-y divide-stone-100 overflow-hidden rounded-lg border border-stone-200 bg-white">
                {group.docs.map((d) => (
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
                        <span className="font-mono">{d.slug}</span>
                        {d.lastUpdated && <span>· updated {d.lastUpdated}</span>}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
