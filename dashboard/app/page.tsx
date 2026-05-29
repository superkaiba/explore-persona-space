/**
 * / — Overview (PUBLIC landing).
 *
 * Public, read-only entry point to the dashboard. Renders the two living
 * orientation docs (open questions + project summary) through the shared
 * MarkdownDoc keystone in `public` mode (sanitized, comments + Ask-Claude
 * disabled), plus a compact "Recent activity" strip linking out to the
 * canonical homes of the latest completed clean-results (/results/[id]) and
 * the most-recently-touched docs (/docs/[slug]).
 *
 * The task list that used to live here now lives at /tasks (read-gated).
 *
 * All disk reads happen server-side via the read-only libs lib/docs.ts +
 * lib/tasks.ts. force-dynamic so the page reflects live tasks/ + docs/ on
 * every request (mirrors every other disk-reading route).
 */
import Link from "next/link";
import { getDoc, listDocs } from "@/lib/docs";
import { listAllTasks } from "@/lib/tasks";
import { MarkdownDoc } from "@/components/MarkdownDoc";

export const dynamic = "force-dynamic";

const OVERVIEW_DOC_SLUGS = ["open_questions", "SUMMARY"] as const;

type ActivityItem = {
  key: string;
  href: string;
  label: string; // short kind label, e.g. "Result" / "Doc"
  title: string;
  meta: string | null; // secondary line (date / id)
  sortKey: number; // higher = more recent
};

/**
 * Latest completed clean-results that the user has classified as `useful`.
 *
 * Reads the authoritative `classification` frontmatter field (NOT a prose
 * regex). Public Results predicate = status `completed` + has_clean_result +
 * classification `useful`. Each links to its canonical home /results/[id].
 */
function recentResults(limit: number): ActivityItem[] {
  return listAllTasks()
    .filter(
      (t) =>
        t.status === "completed" &&
        t.hasCleanResult &&
        t.classification === "useful",
    )
    .slice(0, limit) // listAllTasks() is already sorted by id descending
    .map((t) => ({
      key: `result-${t.id}`,
      href: `/results/${t.id}`,
      label: "Result",
      title: t.title || `#${t.id}`,
      meta: `#${t.id}`,
      sortKey: t.id,
    }));
}

/**
 * Most-recently-touched docs (by lastUpdated), excluding the two orientation
 * docs already rendered in full on this page. Each links to /docs/[slug].
 */
function recentDocs(limit: number): ActivityItem[] {
  const skip = new Set<string>(OVERVIEW_DOC_SLUGS);
  return listDocs()
    .filter((d) => !skip.has(d.slug) && d.lastUpdated)
    .map((d) => ({
      key: `doc-${d.slug}`,
      href: `/docs/${d.slug}`,
      label: "Doc",
      title: d.title,
      meta: d.lastUpdated ? `updated ${d.lastUpdated}` : null,
      // lastUpdated is an ISO date string (YYYY-MM-DD); Date.parse gives a
      // comparable epoch. Non-parseable values sort to the bottom.
      sortKey: d.lastUpdated ? Date.parse(d.lastUpdated) || 0 : 0,
    }))
    .sort((a, b) => b.sortKey - a.sortKey)
    .slice(0, limit);
}

export default async function Overview() {
  const docs = OVERVIEW_DOC_SLUGS.map((slug) => getDoc(slug)).filter(
    (d): d is NonNullable<ReturnType<typeof getDoc>> => d != null,
  );
  const results = recentResults(10);
  const recentDocsList = recentDocs(6);

  return (
    <div className="space-y-10">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Explore Persona Space
        </h1>
        <p className="mt-1 text-sm text-stone-600">
          Characterizing persona representations in language models — geometry,
          localization, propagation, axis origins, defense against emergent
          misalignment.
        </p>
      </header>

      <RecentActivity results={results} docs={recentDocsList} />

      {docs.map((doc) => (
        <section key={doc.slug} className="space-y-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-xl font-semibold tracking-tight">{doc.title}</h2>
            <Link
              href={`/docs/${doc.slug}`}
              className="text-xs text-stone-500 hover:text-stone-800"
            >
              open in Docs →
            </Link>
          </div>
          <div className="rounded-lg border border-stone-200 bg-white px-4 py-4 sm:px-6 sm:py-6">
            <MarkdownDoc
              body={doc.body}
              public
              showToc
              docId={`overview-${doc.slug}`}
            />
          </div>
        </section>
      ))}
    </div>
  );
}

function RecentActivity({
  results,
  docs,
}: {
  results: ActivityItem[];
  docs: ActivityItem[];
}) {
  if (results.length === 0 && docs.length === 0) return null;
  return (
    <section className="space-y-3">
      <h2 className="text-xl font-semibold tracking-tight">Recent activity</h2>
      <div className="grid gap-4 sm:grid-cols-2">
        <ActivityColumn
          heading="Latest results"
          emptyLabel="No promoted clean results yet."
          items={results}
          allHref="/results"
          allLabel="All results →"
        />
        <ActivityColumn
          heading="Recently updated docs"
          emptyLabel="No docs yet."
          items={docs}
          allHref="/docs"
          allLabel="All docs →"
        />
      </div>
    </section>
  );
}

function ActivityColumn({
  heading,
  emptyLabel,
  items,
  allHref,
  allLabel,
}: {
  heading: string;
  emptyLabel: string;
  items: ActivityItem[];
  allHref: string;
  allLabel: string;
}) {
  return (
    <div className="overflow-hidden rounded-lg border border-stone-200 bg-white">
      <div className="flex items-center justify-between border-b border-stone-100 px-4 py-2.5">
        <span className="text-sm font-medium text-stone-700">{heading}</span>
        <Link href={allHref} className="text-xs text-stone-500 hover:text-stone-800">
          {allLabel}
        </Link>
      </div>
      {items.length === 0 ? (
        <p className="px-4 py-4 text-sm text-stone-500">{emptyLabel}</p>
      ) : (
        <ul className="divide-y divide-stone-100">
          {items.map((item) => (
            <li key={item.key}>
              <Link
                href={item.href}
                className="flex items-baseline gap-3 px-4 py-2.5 hover:bg-stone-50"
              >
                <span className="text-sm leading-snug text-stone-900">
                  {item.title}
                </span>
                {item.meta && (
                  <span className="ml-auto whitespace-nowrap font-mono text-xs text-stone-400">
                    {item.meta}
                  </span>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
