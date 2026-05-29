/**
 * /updates — the consolidated, reverse-chronological POINTER feed that
 * merges what used to live across /updates and /log into one timeline.
 *
 * Aggregates two sources (see lib/logs#listUpdatesFeed):
 *   - completed clean-results  -> pointer to /results/<id>
 *   - dated docs               -> pointer to /docs/<slug>
 *     (docs/mentor_updates, logs/daily, logs/weekly)
 *
 * Each card is a POINTER to its canonical home; the feed never re-renders
 * the full body. Filtering (category chips / date range / search) lives in
 * the client <UpdatesFeed> shell with URL-synced state so the view is
 * shareable.
 *
 * Read-gated. The proxy matcher gates /updates at the edge; this server-side
 * `requireSessionAuth()` is defense-in-depth (keeps the component honest if
 * the proxy is bypassed in local dev).
 *
 * Server component: reads from disk, force-dynamic. The /log route it
 * subsumes is retired (the integration step adds the /log -> /updates
 * redirect in next.config).
 */
import { requireSessionAuth } from "@/lib/auth";
import { listUpdatesFeed } from "@/lib/logs";
import { UpdatesFeed } from "./UpdatesFeed";

export const dynamic = "force-dynamic";

type SearchParams = {
  cat?: string;
  from?: string;
  to?: string;
  q?: string;
};

export default async function UpdatesPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  await requireSessionAuth();

  const sp = await searchParams;
  const from = typeof sp.from === "string" ? sp.from : undefined;
  const to = typeof sp.to === "string" ? sp.to : undefined;
  const q = typeof sp.q === "string" ? sp.q : "";

  const items = listUpdatesFeed();

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Updates</h1>
        <p className="mt-1 text-sm text-stone-600">
          One timeline of recent activity. Completed results plus dated mentor
          updates, daily, and weekly notes — each card links to its canonical
          page.
        </p>
      </header>

      <UpdatesFeed
        items={items}
        initialChips={{
          cat: sp.cat ?? null,
          from,
          to,
          q,
        }}
      />
    </div>
  );
}
