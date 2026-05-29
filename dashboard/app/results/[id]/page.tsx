/**
 * /results/[id] — public detail view for one promoted clean result.
 *
 * Public surface. Renders the task body through the shared <MarkdownDoc>
 * keystone in `public` mode: comment writes + Ask-Claude are disabled, the
 * render pipeline is sanitized, and legacy Sagan-card bodies (carrying the
 * `<!-- legacy-sagan-card -->` sentinel, detected by lib/results) take the
 * sanitized trusted-HTML path. A hand-crafted URL for a non-public task
 * 404s: getPublicResult re-applies the completed+`useful` predicate.
 *
 * The body source carries its own `# <title>` H1 (clean-result spec), which
 * MarkdownDoc renders and the TOC picks up as the first entry — so the page
 * header stays a compact breadcrumb + meta strip, no duplicate title.
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { MarkdownDoc } from "@/components/MarkdownDoc";
import { getPublicResult, type ResultConfidence } from "@/lib/results";

export const dynamic = "force-dynamic";

const CONFIDENCE_STYLE: Record<
  Exclude<ResultConfidence, null>,
  { chip: string; dot: string }
> = {
  HIGH: { chip: "bg-emerald-50 text-emerald-700 border-emerald-200", dot: "bg-emerald-500" },
  MODERATE: { chip: "bg-amber-50 text-amber-800 border-amber-200", dot: "bg-amber-500" },
  LOW: { chip: "bg-stone-100 text-stone-600 border-stone-200", dot: "bg-stone-400" },
};

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export default async function ResultDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) notFound();

  const result = getPublicResult(id);
  if (!result) notFound();

  const date = formatDate(result.date);

  return (
    <article className="space-y-6">
      <header className="space-y-3">
        <div className="flex flex-wrap items-baseline gap-3 text-sm text-stone-500">
          <Link href="/results" className="hover:text-stone-800">
            ← All results
          </Link>
          <span aria-hidden>·</span>
          <span className="font-mono">#{result.id}</span>
          {date && (
            <>
              <span aria-hidden>·</span>
              <time>{date}</time>
            </>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          {result.confidence && <ConfidenceBadge confidence={result.confidence} />}
          {result.tags.map((t) => (
            <span
              key={t}
              className="rounded bg-stone-100 px-2 py-0.5 text-stone-700"
            >
              #{t}
            </span>
          ))}
        </div>
      </header>

      <MarkdownDoc
        body={result.body}
        isLegacyHtml={result.isLegacyHtml}
        docId={id}
        showToc
        public
      />
    </article>
  );
}

function ConfidenceBadge({
  confidence,
}: {
  confidence: Exclude<ResultConfidence, null>;
}) {
  const style = CONFIDENCE_STYLE[confidence];
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 font-medium ${style.chip}`}
    >
      <span className={`mr-1.5 inline-block h-2 w-2 rounded-full ${style.dot}`} />
      {confidence.charAt(0) + confidence.slice(1).toLowerCase()} confidence
    </span>
  );
}
