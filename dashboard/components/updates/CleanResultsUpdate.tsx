import Link from "next/link";
import { CalendarDays, ChevronLeft, ChevronRight, MessageCircle } from "lucide-react";
import {
  asDate,
  type CleanResult,
  type CleanResultDateField,
  dayKey,
  formatDay,
  formatShortDate,
  groupResultsByDay,
  resultCounts,
} from "@/lib/update-results";
import {
  ClaudeAskComposer,
  MentorClaudePanel,
} from "@/components/updates/MentorClaudePanel";
import {
  InteractiveResultCard,
  InteractiveResultRow,
} from "@/components/updates/InteractiveResult";
import { cn } from "@/lib/utils";

export function DailyCleanResultsUpdate({
  results,
  archive,
  selectedDate,
  generatedAt,
  internal = false,
  dayPath = "/timeline/today",
  weekPath = "/timeline/week",
  showInternalLink = true,
}: {
  results: CleanResult[];
  archive: CleanResult[];
  selectedDate: Date;
  generatedAt: Date;
  internal?: boolean;
  dayPath?: string;
  weekPath?: string;
  showInternalLink?: boolean;
}) {
  const selectedKey = dayKey(selectedDate);
  const archiveGroups = groupResultsByDay(archive);
  const previousGroup = archiveGroups.find((group) => group.date < selectedDate);
  const newerGroup = archiveGroups
    .slice()
    .reverse()
    .find((group) => group.date > selectedDate);
  const context = dailyContext(selectedDate, results);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-5 py-6 md:px-8">
        <header className="mb-5 flex flex-wrap items-start justify-between gap-3 border-b border-border pb-4">
          <div>
            <h1 className="text-[22px] font-semibold tracking-tight text-fg">
              Daily update
            </h1>
            <p className="mt-1 text-[12px] text-muted">
              Results finalized on {formatDay(selectedDate)}
              {selectedKey === dayKey(generatedAt) ? "" : "."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Link
              href={internal ? `${weekPath}?internal=1` : weekPath}
              className="rounded-md border border-border bg-panel px-3 py-1.5 text-[12px] text-muted hover:bg-subtle hover:text-fg"
            >
              Weekly
            </Link>
            {showInternalLink && !internal && (
              <Link
                href={`${dayPath}?date=${selectedKey}&internal=1`}
                className="rounded-md border border-border bg-panel px-3 py-1.5 text-[12px] text-muted hover:bg-subtle hover:text-fg"
              >
                Internal
              </Link>
            )}
          </div>
        </header>

        <ClaudeAskComposer
          payload={{
            scopeKind: "global",
            scopeId: `daily-${selectedKey}`,
            sourceLabel: "Global",
            scopeTitle: `Daily update - ${formatDay(selectedDate)}`,
            contextMd: context,
            suggestedQuestion: "What should I take away from today's results?",
          }}
          placeholder="Ask Claude Code to inspect today's results..."
          className="mb-5"
        />

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_260px]">
          <main className="min-w-0">
            <DayStepper
              selectedDate={selectedDate}
              previousKey={previousGroup?.key ?? null}
              nextKey={newerGroup?.key ?? null}
              internal={internal}
              dayPath={dayPath}
            />

            {results.length === 0 ? (
              <p className="mt-4 rounded-lg border border-dashed border-border bg-panel p-4 text-[13px] text-muted">
                No clean results were finalized on this day.
              </p>
            ) : (
              <div className="mt-4 flex flex-col gap-3">
                {results.map((result) => (
                  <InteractiveResultCard
                    key={result.id}
                    result={clientResult(result)}
                    internal={internal}
                  />
                ))}
              </div>
            )}
          </main>

          <aside className="min-w-0">
            <ArchiveNav
              groups={archiveGroups}
              activeKey={selectedKey}
              internal={internal}
              dayPath={dayPath}
              selectedDate={selectedDate}
            />
          </aside>
        </div>

        <MentorClaudePanel sessionId={`daily-${selectedKey}`} baseContextMd={context} />
      </div>
    </div>
  );
}

export function CleanResultsLogUpdate({
  results,
  generatedAt,
  internal = false,
  weekPath = "/timeline/week",
  showWeeklyLink = true,
  showInternalLink = true,
  dateField = "updatedAt",
  description,
}: {
  results: CleanResult[];
  generatedAt: Date;
  internal?: boolean;
  weekPath?: string;
  showWeeklyLink?: boolean;
  showInternalLink?: boolean;
  dateField?: CleanResultDateField;
  description?: string;
}) {
  const sortedResults = results
    .slice()
    .sort((a, b) => asDate(b[dateField]).getTime() - asDate(a[dateField]).getTime());
  const groups = groupResultsByDay(sortedResults, dateField);
  const context = logContext(sortedResults, dateField);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-5 py-6 md:px-8">
        <header className="mb-5 flex flex-wrap items-start justify-between gap-3 border-b border-border pb-4">
          <div>
            <h1 className="text-[22px] font-semibold tracking-tight text-fg">
              Results log
            </h1>
            <p className="mt-1 text-[12px] text-muted">
              {description ??
                (dateField === "createdAt"
                  ? "GitHub Useful and Not useful columns, newest issue creation first."
                  : "Clean results, most recent first.")}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {showWeeklyLink && (
              <Link
                href={internal ? `${weekPath}?internal=1` : weekPath}
                className="rounded-md border border-border bg-panel px-3 py-1.5 text-[12px] text-muted hover:bg-subtle hover:text-fg"
              >
                Weekly
              </Link>
            )}
            {showInternalLink && !internal && (
              <Link
                href="/timeline/today?internal=1"
                className="rounded-md border border-border bg-panel px-3 py-1.5 text-[12px] text-muted hover:bg-subtle hover:text-fg"
              >
                Internal
              </Link>
            )}
          </div>
        </header>

        <ClaudeChatExplainer />

        <ClaudeAskComposer
          payload={{
            scopeKind: "global",
            scopeId: `log-${dayKey(generatedAt)}`,
            sourceLabel: "Global",
            scopeTitle: "Results log",
            contextMd: context,
            suggestedQuestion: "What should I take away from the recent clean results?",
          }}
          placeholder="Ask Claude Code to inspect these results..."
          className="mb-5"
        />

        {groups.length === 0 ? (
          <p className="rounded-lg border border-dashed border-border bg-panel p-4 text-[13px] text-muted">
            No clean results in this window.
          </p>
        ) : (
          <main className="flex flex-col gap-7">
            {groups.map((group) => {
              const counts = resultCounts(group.results);
              return (
                <section key={group.key} className="min-w-0">
                  <div className="sticky top-0 z-20 -mx-5 flex items-center justify-between gap-3 border-y border-border bg-canvas/95 px-5 py-2 backdrop-blur md:-mx-8 md:px-8">
                    <div>
                      <h2 className="text-[13px] font-semibold text-fg">{group.label}</h2>
                      <div className="font-mono text-[10px] text-muted">{group.key}</div>
                    </div>
                    <div className="text-right text-[11px] text-muted">
                      <div>
                        {counts.total} {counts.total === 1 ? "result" : "results"}
                      </div>
                      <div>
                        {counts.useful} useful, {counts.notUseful} not useful
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-col gap-3">
                    {group.results.map((result) => (
                      <InteractiveResultCard
                        key={result.id}
                        result={clientResult(result)}
                        internal={internal}
                        dateField={dateField}
                      />
                    ))}
                  </div>
                </section>
              );
            })}
          </main>
        )}

        <MentorClaudePanel sessionId={`log-${dayKey(generatedAt)}`} baseContextMd={context} />
      </div>
    </div>
  );
}

export function WeeklyCleanResultsUpdate({
  week,
  recent,
  generatedAt,
  weekStart,
  internal = false,
  dayPath = "/timeline/today",
  dateField = "updatedAt",
}: {
  week: CleanResult[];
  recent: CleanResult[];
  generatedAt: Date;
  weekStart: Date;
  internal?: boolean;
  dayPath?: string;
  dateField?: CleanResultDateField;
}) {
  const counts = resultCounts(week);
  const sortedWeek = week
    .slice()
    .sort((a, b) => asDate(b[dateField]).getTime() - asDate(a[dateField]).getTime());
  const sortedRecent = recent
    .slice()
    .sort((a, b) => asDate(b[dateField]).getTime() - asDate(a[dateField]).getTime());
  const groups = groupResultsByDay(sortedWeek, dateField);
  const archiveGroups = groupResultsByDay(sortedRecent, dateField);
  const context = weeklyContext(weekStart, generatedAt, sortedWeek, dateField);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-5 py-6 md:px-8">
        <header className="mb-5 flex flex-wrap items-start justify-between gap-3 border-b border-border pb-4">
          <div>
            <h1 className="text-[22px] font-semibold tracking-tight text-fg">
              Weekly update
            </h1>
            <p className="mt-1 text-[12px] text-muted">
              Results from {formatShortDate(weekStart)} to {formatShortDate(generatedAt)}.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Link
              href={internal ? `${dayPath}?internal=1` : dayPath}
              className="rounded-md border border-border bg-panel px-3 py-1.5 text-[12px] text-muted hover:bg-subtle hover:text-fg"
            >
              Daily
            </Link>
          </div>
        </header>

        <ClaudeAskComposer
          payload={{
            scopeKind: "global",
            scopeId: "weekly-current",
            sourceLabel: "Global",
            scopeTitle: "Weekly update",
            contextMd: context,
            suggestedQuestion: "What are the main takeaways from this week's results?",
          }}
          placeholder="Ask Claude Code to inspect this week's results..."
          className="mb-4"
        />

        <div className="mb-4 grid grid-cols-3 gap-2 text-[12px]">
          <Summary value={counts.total} label="results" />
          <Summary value={counts.useful} label="useful" />
          <Summary value={counts.notUseful} label="not useful" />
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_260px]">
          <main className="min-w-0">
            {groups.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border bg-panel p-4 text-[13px] text-muted">
                No clean results were finalized this week.
              </p>
            ) : (
              <div className="flex flex-col gap-6">
                {groups.map((group) => (
                  <section key={group.key}>
                    <div className="mb-2 flex items-center justify-between border-b border-border pb-2">
                      <h2 className="text-[13px] font-semibold text-fg">{group.label}</h2>
                      <Link
                        href={`${dayPath}?date=${group.key}${internal ? "&internal=1" : ""}`}
                        className="text-[11px] text-muted hover:text-fg"
                      >
                        Open day
                      </Link>
                    </div>
                    <div className="flex flex-col gap-2">
                      {group.results.map((result) => (
                        <InteractiveResultRow
                          key={result.id}
                          result={clientResult(result)}
                          internal={internal}
                          dateField={dateField}
                        />
                      ))}
                    </div>
                  </section>
                ))}
              </div>
            )}
          </main>

          <aside className="min-w-0">
            <ArchiveNav
              groups={archiveGroups}
              activeKey=""
              internal={internal}
              dayPath={dayPath}
              selectedDate={generatedAt}
            />
          </aside>
        </div>

        <MentorClaudePanel sessionId="weekly-current" baseContextMd={context} />
      </div>
    </div>
  );
}

function DayStepper({
  selectedDate,
  previousKey,
  nextKey,
  internal,
  dayPath,
}: {
  selectedDate: Date;
  previousKey: string | null;
  nextKey: string | null;
  internal: boolean;
  dayPath: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-panel px-3 py-2">
      <DayLink
        dayKeyValue={previousKey}
        internal={internal}
        direction="prev"
        dayPath={dayPath}
      />
      <div className="text-center">
        <div className="text-[13px] font-semibold text-fg">{formatDay(selectedDate)}</div>
        <div className="font-mono text-[10px] text-muted">{dayKey(selectedDate)}</div>
      </div>
      <DayLink
        dayKeyValue={nextKey}
        internal={internal}
        direction="next"
        dayPath={dayPath}
      />
    </div>
  );
}

function DayLink({
  dayKeyValue,
  internal,
  direction,
  dayPath,
}: {
  dayKeyValue: string | null;
  internal: boolean;
  direction: "prev" | "next";
  dayPath: string;
}) {
  const content =
    direction === "prev" ? (
      <>
        <ChevronLeft className="h-4 w-4" />
        Older
      </>
    ) : (
      <>
        Newer
        <ChevronRight className="h-4 w-4" />
      </>
    );

  if (!dayKeyValue) {
    return (
      <span className="inline-flex items-center gap-1 text-[12px] text-muted/45">
        {content}
      </span>
    );
  }

  return (
    <Link
      href={`${dayPath}?date=${dayKeyValue}${internal ? "&internal=1" : ""}`}
      className="inline-flex items-center gap-1 text-[12px] text-muted hover:text-fg"
    >
      {content}
    </Link>
  );
}

type ArchiveGroup = {
  key: string;
  label: string;
  date: Date;
  results: CleanResult[];
};

function ArchiveNav({
  groups,
  activeKey,
  internal,
  dayPath,
  selectedDate,
}: {
  groups: ArchiveGroup[];
  activeKey: string;
  internal: boolean;
  dayPath: string;
  selectedDate: Date;
}) {
  const selectedKey = activeKey || dayKey(selectedDate);
  const months = archiveMonths(groups.slice(0, 60));

  if (groups.length === 0) {
    return (
      <section>
        <h2 className="mb-3 text-[13px] font-semibold text-fg">Browse dates</h2>
        <DateJumpForm
          selectedKey={selectedKey}
          internal={internal}
          dayPath={dayPath}
        />
        <p className="rounded-lg border border-dashed border-border p-3 text-[12px] text-muted">
          No past updates in this window.
        </p>
      </section>
    );
  }

  return (
    <section>
      <div className="mb-3 flex items-center gap-2">
        <CalendarDays className="h-4 w-4 text-muted" />
        <h2 className="text-[13px] font-semibold text-fg">Browse dates</h2>
      </div>

      <DateJumpForm selectedKey={selectedKey} internal={internal} dayPath={dayPath} />

      <div className="mt-3 max-h-[52dvh] overflow-y-auto rounded-lg border border-border bg-panel">
        {months.map((month) => (
          <div key={month.key}>
            <div className="sticky top-0 z-10 border-b border-border bg-panel px-3 py-2 text-[11px] font-medium text-muted">
              {month.label}
            </div>
            <div className="divide-y divide-border">
              {month.groups.map((group) => {
                const counts = resultCounts(group.results);
                return (
                  <Link
                    key={group.key}
                    href={`${dayPath}?date=${group.key}${internal ? "&internal=1" : ""}`}
                    className={cn(
                      "block px-3 py-2.5 transition-colors hover:bg-subtle/70",
                      selectedKey === group.key && "bg-subtle",
                    )}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-[12px] font-medium text-fg">{group.label}</div>
                      <div className="font-mono text-[11px] text-muted">
                        {counts.total} {counts.total === 1 ? "result" : "results"}
                      </div>
                    </div>
                    <div className="mt-1 text-[11px] text-muted">
                      {counts.useful} useful, {counts.notUseful} not useful
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function DateJumpForm({
  selectedKey,
  internal,
  dayPath,
}: {
  selectedKey: string;
  internal: boolean;
  dayPath: string;
}) {
  return (
    <form action={dayPath} method="get" className="rounded-lg border border-border bg-panel p-3">
      {internal && <input type="hidden" name="internal" value="1" />}
      <label htmlFor="daily-update-date" className="text-[11px] font-medium text-muted">
        Jump to date
      </label>
      <div className="mt-2 flex gap-2">
        <input
          id="daily-update-date"
          type="date"
          name="date"
          defaultValue={selectedKey}
          className="min-w-0 flex-1 rounded-md border border-border bg-canvas px-2 py-1.5 text-[12px] text-fg outline-none focus:border-border-strong"
        />
        <button
          type="submit"
          className="rounded-md border border-border bg-subtle px-2.5 py-1.5 text-[12px] text-fg hover:bg-raised"
        >
          Open
        </button>
      </div>
      <Link
        href={internal ? `${dayPath}?internal=1` : dayPath}
        className="mt-2 inline-flex text-[11px] text-muted hover:text-fg"
      >
        Today
      </Link>
    </form>
  );
}

function archiveMonths(groups: ArchiveGroup[]) {
  const months: Array<{ key: string; label: string; groups: ArchiveGroup[] }> = [];

  for (const group of groups) {
    const key = `${group.date.getFullYear()}-${String(group.date.getMonth() + 1).padStart(
      2,
      "0",
    )}`;
    const label = group.date.toLocaleDateString(undefined, {
      month: "long",
      year: "numeric",
    });
    const existing = months[months.length - 1];
    if (existing?.key === key) {
      existing.groups.push(group);
    } else {
      months.push({ key, label, groups: [group] });
    }
  }

  return months;
}

function Summary({ value, label }: { value: number; label: string }) {
  return (
    <div className="rounded-lg border border-border bg-panel p-3">
      <div className="font-mono text-[18px] leading-none text-fg">{value}</div>
      <div className="mt-1 text-[11px] text-muted">{label}</div>
    </div>
  );
}

function ClaudeChatExplainer() {
  return (
    <section className="mb-4 rounded-lg border border-border bg-panel p-3">
      <div className="flex gap-3">
        <MessageCircle className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
        <div className="min-w-0 text-[12px] leading-relaxed text-fg-soft">
          <div className="font-medium text-fg">Claude Code chats</div>
          <p className="mt-1 max-w-3xl">
            Use the top box for the whole log, or a result&apos;s chat button for that
            result. Each chat starts a Claude Code sidecar that can inspect the VM,
            database, GitHub issue, and linked artifacts. Result and global chats stay
            in one tabbed chat box; closed chats keep their tabs and messages when
            reopened.
          </p>
        </div>
      </div>
    </section>
  );
}

function clientResult(result: CleanResult): CleanResult {
  return {
    ...result,
    createdAt: asDate(result.createdAt).toISOString(),
    updatedAt: asDate(result.updatedAt).toISOString(),
  };
}

function dailyContext(date: Date, results: CleanResult[]) {
  const lines = results.map((result, index) => {
    return [
      `${index + 1}. ${result.title}`,
      `   Claim ID: ${result.id}`,
      result.githubIssueNumber == null
        ? null
        : `   GitHub issue: https://github.com/superkaiba/explore-persona-space/issues/${result.githubIssueNumber}`,
      `   Classification: ${result.useful ? "useful" : "not useful"}`,
      `   Confidence: ${result.confidence ?? "not set"}`,
      result.excerpt ? `   Excerpt: ${result.excerpt}` : null,
      `   Dashboard URL: https://dashboard.superkaiba.com${result.href}`,
    ]
      .filter(Boolean)
      .join("\n");
  });
  return [`Daily update for ${formatDay(date)}.`, "", ...lines].join("\n");
}

function weeklyContext(
  start: Date,
  end: Date,
  results: CleanResult[],
  dateField: CleanResultDateField = "updatedAt",
) {
  return [
    `Weekly update from ${formatShortDate(start)} to ${formatShortDate(end)}.`,
    "",
    ...results.map((result, index) =>
      [
        `${index + 1}. ${result.title} (${dayKey(result[dateField])})`,
        `   Claim ID: ${result.id}`,
        result.githubIssueNumber == null
          ? null
          : `   GitHub issue: https://github.com/superkaiba/explore-persona-space/issues/${result.githubIssueNumber}`,
      ]
        .filter(Boolean)
        .join("\n"),
    ),
  ].join("\n");
}

function logContext(
  results: CleanResult[],
  dateField: CleanResultDateField = "updatedAt",
) {
  return [
    dateField === "createdAt"
      ? "Recent clean results log from GitHub Useful and Not useful columns, newest issue creation first."
      : "Recent clean results log, most recent first.",
    "",
    ...results.map((result, index) =>
      [
        `${index + 1}. ${result.title} (${dayKey(result[dateField])})`,
        `   Claim ID: ${result.id}`,
        result.githubIssueNumber == null
          ? null
          : `   GitHub issue: https://github.com/superkaiba/explore-persona-space/issues/${result.githubIssueNumber}`,
        `   Classification: ${result.useful ? "useful" : "not useful"}`,
        `   Confidence: ${result.confidence ?? "not set"}`,
        result.excerpt ? `   Excerpt: ${result.excerpt}` : null,
      ]
        .filter(Boolean)
        .join("\n"),
    ),
  ].join("\n");
}
