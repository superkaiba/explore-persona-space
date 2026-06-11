"use client";

/**
 * Client shell for /sessions. Two responsibilities:
 *
 *   1. Auto-refresh the server data every 30s via router.refresh(). The
 *      parent page is `force-dynamic`, so each refresh re-reads the
 *      cache file from disk and re-renders the table without losing
 *      scroll position.
 *
 *   2. Tick a local clock every 30s so the "x minutes ago" relative
 *      timestamps stay current between server refreshes.
 *
 * The table itself is presentational: every prop comes from the server
 * page, sorted in place. Sessions with an `error` field render the
 * error visibly in red (per spec — "not blank, not hidden").
 */
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { STATUS_LABELS, type Status } from "@/lib/repo";

const POLL_INTERVAL_MS = 30_000;
const CLOCK_TICK_MS = 30_000;

export type SessionRowView = {
  sessionId: string;
  issue: number | null;
  issueResolved: boolean;
  issueTitle: string | null;
  status: string | null;
  dir: string | null;
  live: boolean | null;
  pid: number | null;
  summary: string | null;
  summaryModel: string | null;
  summaryTs: string | null;
  /** "self" = byte-identical to phone title; "llm" = Haiku-summarized;
   *  null = legacy entry or no summary. Surfaced as a small tag next to
   *  the summary so the user can distinguish a self-reported canonical
   *  string from an LLM paraphrase. */
  source: "self" | "llm" | null;
  lastActivityTs: string | null;
  error: string | null;
};

export function SessionsTable({
  rows,
  updatedAt,
  readError,
}: {
  rows: SessionRowView[];
  updatedAt: string | null;
  readError: string | null;
}) {
  const router = useRouter();
  const [now, setNow] = useState<number>(() => Date.now());

  // Server refresh (re-reads the cache file) and local clock tick (keeps
  // relative timestamps fresh between refreshes) run on independent
  // intervals so a slow server round-trip never freezes the timestamps.
  useEffect(() => {
    const refresh = window.setInterval(() => router.refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(refresh);
  }, [router]);

  useEffect(() => {
    const tick = window.setInterval(() => setNow(Date.now()), CLOCK_TICK_MS);
    return () => window.clearInterval(tick);
  }, []);

  return (
    <div className="space-y-3">
      <StatusBar
        updatedAt={updatedAt}
        readError={readError}
        rowCount={rows.length}
        now={now}
      />

      {readError ? (
        <ErrorPanel message={readError} />
      ) : rows.length === 0 ? (
        <EmptyPanel hasFile={updatedAt !== null} />
      ) : (
        <SessionsList rows={rows} now={now} />
      )}
    </div>
  );
}

function StatusBar({
  updatedAt,
  readError,
  rowCount,
  now,
}: {
  updatedAt: string | null;
  readError: string | null;
  rowCount: number;
  now: number;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-stone-200 bg-white px-3 py-2 text-xs text-stone-500">
      <span>
        {rowCount} session{rowCount === 1 ? "" : "s"} · auto-refresh every 30s
      </span>
      <span>
        {readError ? (
          <span className="text-rose-600">cache read failed</span>
        ) : updatedAt ? (
          <>
            data as of{" "}
            <time dateTime={updatedAt} title={updatedAt}>
              {formatRelative(updatedAt, now)}
            </time>
          </>
        ) : (
          <span className="text-stone-400">no cache file yet</span>
        )}
      </span>
    </div>
  );
}

function EmptyPanel({ hasFile }: { hasFile: boolean }) {
  return (
    <div className="rounded-lg border border-dashed border-stone-300 bg-white px-5 py-10 text-center text-sm text-stone-500">
      <p className="font-medium text-stone-700">No session data yet</p>
      <p className="mt-2 max-w-prose mx-auto">
        {hasFile
          ? "The cache file is present but lists no active sessions. The summarizer cron writes ~/.eps-autonomous/session_progress.json every 5 minutes."
          : "The cache file ~/.eps-autonomous/session_progress.json hasn't been written yet. The summarizer cron writes it every 5 minutes."}
      </p>
    </div>
  );
}

function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-800">
      <p className="font-medium">Cache read failed</p>
      <p className="mt-1 font-mono text-xs text-rose-700">{message}</p>
    </div>
  );
}

function SessionsList({ rows, now }: { rows: SessionRowView[]; now: number }) {
  return (
    <div className="overflow-hidden rounded-lg border border-stone-200 bg-white">
      <table className="w-full text-sm">
        <thead className="border-b border-stone-200 bg-stone-50 text-left text-xs uppercase tracking-wide text-stone-500">
          <tr>
            <th scope="col" className="px-4 py-2 font-medium">
              Issue
            </th>
            <th scope="col" className="px-4 py-2 font-medium">
              Status
            </th>
            <th scope="col" className="px-4 py-2 font-medium">
              Progress
            </th>
            <th scope="col" className="px-4 py-2 font-medium whitespace-nowrap">
              Last activity
            </th>
            <th scope="col" className="px-4 py-2 font-medium whitespace-nowrap">
              Summary age
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-100">
          {rows.map((row) => (
            <SessionRow key={row.sessionId} row={row} now={now} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SessionRow({ row, now }: { row: SessionRowView; now: number }) {
  return (
    <tr className={row.error ? "bg-rose-50/40" : "hover:bg-stone-50"}>
      <td className="px-4 py-3 align-top">
        <IssueCell row={row} />
      </td>
      <td className="px-4 py-3 align-top">
        <StatusBadge status={row.status} />
        {row.live === false && (
          <div className="mt-1 text-[11px] uppercase tracking-wide text-stone-400">
            session not live
          </div>
        )}
      </td>
      <td className="px-4 py-3 align-top">
        <ProgressCell row={row} />
      </td>
      <td className="px-4 py-3 align-top whitespace-nowrap text-xs text-stone-600">
        <RelativeTime ts={row.lastActivityTs} now={now} />
      </td>
      <td className="px-4 py-3 align-top whitespace-nowrap text-xs text-stone-600">
        <RelativeTime ts={row.summaryTs} now={now} />
      </td>
    </tr>
  );
}

function IssueCell({ row }: { row: SessionRowView }) {
  if (row.issue == null) {
    return <span className="font-mono text-xs text-rose-600">no issue</span>;
  }
  if (!row.issueResolved) {
    return (
      <div className="space-y-0.5">
        <span className="font-mono text-sm text-rose-600">#{row.issue}</span>
        <div className="text-[11px] uppercase tracking-wide text-rose-500">
          not found
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-0.5">
      <Link
        href={`/tasks/${row.issue}`}
        className="font-mono text-sm text-stone-700 hover:underline"
      >
        #{row.issue}
      </Link>
      {row.issueTitle && (
        <div className="line-clamp-2 text-xs text-stone-500" title={row.issueTitle}>
          {row.issueTitle}
        </div>
      )}
    </div>
  );
}

function ProgressCell({ row }: { row: SessionRowView }) {
  if (row.error) {
    return (
      <div className="space-y-1">
        <p className="text-sm font-medium text-rose-700">Error</p>
        <p className="text-xs text-rose-700">{row.error}</p>
      </div>
    );
  }
  // Provenance suffix: "self" rows surface a small "self-report" tag
  // (the string is byte-identical to the phone title); "llm" rows show
  // the Haiku model id; legacy / no-source rows show neither. Keeps the
  // user honest about what produced the line.
  const provenance =
    row.source === "self"
      ? " · self-report"
      : row.summaryModel
        ? ` · ${row.summaryModel}`
        : "";
  return (
    <div className="space-y-1">
      <p className="text-sm leading-snug text-stone-800">
        {row.summary || (
          <span className="text-stone-400">no summary yet</span>
        )}
      </p>
      <p className="text-[11px] text-stone-400">
        <span className="font-mono">{row.sessionId.slice(0, 8)}</span>
        {provenance}
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * Status badge — pulls labels from lib/repo when the status is a known
 * lifecycle enum; falls back to the raw string when it isn't (the cache
 * could carry a future status the dashboard hasn't been redeployed for).
 * -------------------------------------------------------------------------- */

function StatusBadge({ status }: { status: string | null }) {
  if (!status) {
    return (
      <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-500">
        unknown
      </span>
    );
  }
  const known = (STATUS_LABELS as Record<string, string>)[status];
  const label = known ?? status;
  return (
    <span
      className={`rounded px-2 py-0.5 text-xs font-medium ${statusColor(status)}`}
      title={status}
    >
      {label}
    </span>
  );
}

function statusColor(status: string): string {
  switch (status as Status) {
    case "running":
    case "followups_running":
      return "bg-emerald-100 text-emerald-800";
    case "interpreting":
    case "reviewing":
    case "verifying":
      return "bg-blue-100 text-blue-800";
    case "planning":
    case "plan_pending":
      return "bg-violet-100 text-violet-800";
    case "approved":
    case "proposed":
      return "bg-stone-100 text-stone-700";
    case "awaiting_promotion":
      return "bg-amber-100 text-amber-800";
    case "completed":
      return "bg-emerald-50 text-emerald-700";
    case "blocked":
      return "bg-rose-100 text-rose-800";
    case "archived":
      return "bg-stone-100 text-stone-500";
    default:
      return "bg-stone-100 text-stone-700";
  }
}

/* -------------------------------------------------------------------------- *
 * Relative-time rendering.
 * -------------------------------------------------------------------------- */

function RelativeTime({ ts, now }: { ts: string | null; now: number }) {
  if (!ts) return <span className="text-stone-400">—</span>;
  return (
    <time dateTime={ts} title={ts}>
      {formatRelative(ts, now)}
    </time>
  );
}

function formatRelative(iso: string, now: number): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return iso;
  const delta = Math.max(0, Math.round((now - t) / 1000));
  if (delta < 5) return "just now";
  if (delta < 60) return `${delta}s ago`;
  const m = Math.round(delta / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}
