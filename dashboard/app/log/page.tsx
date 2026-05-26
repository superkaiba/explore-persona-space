/**
 * /log — chronological feed interleaving daily / weekly / ideation log
 * entries (from `logs/`) with promoted clean-results (from `tasks/`).
 *
 * Server component: reads everything from disk and hands the merged
 * list to the client `<LogFeed>` for chip filtering, search, and
 * URL-state sync.
 *
 * Filter chips, date range, and search all live in the URL so the view
 * is shareable with the mentor.
 */
import { requireSessionAuth } from "@/lib/auth";
import {
  listCleanResults,
  listLogEntries,
  type FeedItem,
  type FeedItemKind,
  type LogEntryKind,
} from "@/lib/logs";
import { LogFeed } from "./LogFeed";

export const dynamic = "force-dynamic";

const ALL_FEED_KINDS: FeedItemKind[] = ["daily", "weekly", "ideation", "clean-result"];

type SearchParams = {
  kind?: string;
  useful?: string;
  from?: string;
  to?: string;
  q?: string;
};

/**
 * Parse the `?kind=` URL chip into a list of `FeedItemKind` values.
 * Returns `null` when no filter is active (no param, or no recognized
 * tokens), which means "show everything". Recognizes `clean-result` in
 * addition to the three `LogEntryKind` values so the page-level
 * fetch-short-circuit can skip the right side.
 */
function parseKinds(raw: string | undefined): FeedItemKind[] | null {
  if (!raw) return null; // null = "no filter, show everything"
  const wanted = new Set(
    raw
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  );
  const out: FeedItemKind[] = [];
  for (const k of ALL_FEED_KINDS) if (wanted.has(k)) out.push(k);
  return out.length === 0 ? null : out;
}

function withinDateRange(dateIso: string, from?: string, to?: string): boolean {
  // Compare as ISO strings (works for `YYYY-MM-DD`). `from` and `to` are
  // inclusive; either may be missing.
  if (from && dateIso < from) return false;
  if (to && dateIso > to) return false;
  return true;
}

export default async function LogPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  // Gate everything behind the same site-password session the rest of
  // the dashboard uses. The proxy middleware also enforces this at the
  // edge, but a defense-in-depth check here keeps the server component
  // honest when the proxy is bypassed (e.g. during local dev).
  const user = await requireSessionAuth();

  const sp = await searchParams;
  const wantKinds = parseKinds(sp.kind);
  const usefulOnly = sp.useful !== "all";  // default = useful only
  const from = typeof sp.from === "string" ? sp.from : undefined;
  const to = typeof sp.to === "string" ? sp.to : undefined;
  const q = typeof sp.q === "string" ? sp.q : "";

  // Decide what to fetch based on the kind filter. If the user explicitly
  // asked for only `clean-result`, skip the log read entirely (and vice
  // versa for clean-results). Spec defaults to "show everything", so
  // when the chip set is null we fetch both.
  const wantLogs = wantKinds === null || wantKinds.some((k) => k !== "clean-result");
  const wantCleanResults = wantKinds === null || wantKinds.includes("clean-result");
  // listLogEntries only knows about the three LogEntryKind values; strip
  // `clean-result` before handing the filter in.
  const logKindsFilter = wantKinds
    ? wantKinds.filter((k): k is LogEntryKind => k !== "clean-result")
    : undefined;

  const [logEntries, cleanResults] = await Promise.all([
    wantLogs
      ? listLogEntries({
          kinds: logKindsFilter,
          includeDrafts: false,
        })
      : Promise.resolve([]),
    wantCleanResults ? listCleanResults({ includeNotUseful: true }) : Promise.resolve([]),
  ]);

  // Server-side filter by date range. Search + useful-only happen in the
  // client so the user can twiddle them without a round-trip.
  const merged: FeedItem[] = [...logEntries, ...cleanResults].filter((item) =>
    withinDateRange(item.date, from, to),
  );

  // Newest first. Both sources are already date-sorted but the merge
  // needs a re-sort.
  merged.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Log</h1>
        <p className="mt-1 text-sm text-stone-600">
          Chronological results timeline. Daily + weekly + ideation entries
          interleaved with promoted clean-results.
        </p>
      </header>

      <LogFeed
        items={merged}
        initialChips={{
          kind: sp.kind ?? null,
          useful: usefulOnly,
          from,
          to,
          q,
        }}
        currentUserEmail={user?.email ?? null}
      />
    </div>
  );
}
