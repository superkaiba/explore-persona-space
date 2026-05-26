export type CleanResultConfidence = "HIGH" | "MODERATE" | "LOW" | null;

export type CleanResult = {
  id: string;
  title: string;
  body: string;
  excerpt: string;
  confidence: CleanResultConfidence;
  useful: boolean;
  githubIssueNumber: number | null;
  createdAt: Date | string;
  updatedAt: Date | string;
  href: string;
};

export type CleanResultDateField = "createdAt" | "updatedAt";

type BodyJson = { kind?: string; text?: unknown } | null;

export function startOfLocalDay(input = new Date()) {
  const d = new Date(input);
  d.setHours(0, 0, 0, 0);
  return d;
}

export function addDays(input: Date, days: number) {
  const d = new Date(input);
  d.setDate(d.getDate() + days);
  return d;
}

export function startOfLocalWeek(input = new Date()) {
  const d = startOfLocalDay(input);
  const day = d.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  return addDays(d, diff);
}

export function asDate(value: Date | string) {
  return value instanceof Date ? value : new Date(value);
}

export function dayKey(value: Date | string) {
  const d = asDate(value);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function parseDayKey(value: string | null | undefined) {
  if (!value || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split("-").map(Number);
  const d = new Date(year, month - 1, day);
  if (
    d.getFullYear() !== year ||
    d.getMonth() !== month - 1 ||
    d.getDate() !== day
  ) {
    return null;
  }
  return startOfLocalDay(d);
}

export function formatDay(value: Date | string) {
  return asDate(value).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

export function formatShortDate(value: Date | string) {
  return asDate(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function formatTime(value: Date | string) {
  return asDate(value).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function bodyText(bodyJson: unknown) {
  const body = bodyJson as BodyJson;
  return typeof body?.text === "string" ? body.text : "";
}

export function isUsefulCleanResult({
  title,
  body,
  confidence,
}: {
  title: string;
  body: string;
  confidence: CleanResultConfidence;
}) {
  const haystack = `${title}\n${body}`.toLowerCase();
  if (/\bnot[- ]useful\b/.test(haystack)) return false;
  if (/\buseful\b/.test(haystack)) return true;
  return confidence !== "LOW";
}

export function markdownExcerpt(markdown: string, maxLength = 260) {
  const plain = markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/[#>*_`~|-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (plain.length <= maxLength) return plain;
  const clipped = plain.slice(0, maxLength);
  const lastSpace = clipped.lastIndexOf(" ");
  return `${clipped.slice(0, lastSpace > 120 ? lastSpace : maxLength).trim()}...`;
}

export function cleanResultFromClaim(row: {
  id: string;
  title: string;
  confidence: CleanResultConfidence;
  githubIssueNumber: number | null;
  bodyJson: unknown;
  createdAt: Date | string;
  updatedAt: Date | string;
}): CleanResult {
  const body = bodyText(row.bodyJson);
  return {
    id: row.id,
    title: row.title,
    body,
    excerpt: markdownExcerpt(body),
    confidence: row.confidence,
    useful: isUsefulCleanResult({
      title: row.title,
      body,
      confidence: row.confidence,
    }),
    githubIssueNumber: row.githubIssueNumber,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
    href: `/claim/${row.id}`,
  };
}

export function groupResultsByDay(
  results: CleanResult[],
  dateField: CleanResultDateField = "updatedAt",
) {
  const groups: Array<{
    key: string;
    label: string;
    date: Date;
    results: CleanResult[];
  }> = [];

  for (const result of results) {
    const dateValue = result[dateField];
    const key = dayKey(dateValue);
    const last = groups[groups.length - 1];
    if (last?.key === key) {
      last.results.push(result);
      continue;
    }
    groups.push({
      key,
      label: formatDay(dateValue),
      date: startOfLocalDay(asDate(dateValue)),
      results: [result],
    });
  }

  return groups;
}

export function resultCounts(results: CleanResult[]) {
  return {
    total: results.length,
    useful: results.filter((r) => r.useful).length,
    notUseful: results.filter((r) => !r.useful).length,
    high: results.filter((r) => r.confidence === "HIGH").length,
  };
}
