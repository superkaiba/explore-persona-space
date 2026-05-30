/**
 * /results/[id] — public detail view for one promoted clean result.
 *
 * Public surface. Renders the task body through the <TaskCommentBody> shell
 * (which renders the body via the shared <MarkdownDoc> keystone AND mounts the
 * anchored-comment rail). The render pipeline is sanitized, and legacy
 * Sagan-card bodies (carrying the `<!-- legacy-sagan-card -->` sentinel,
 * detected by lib/results) take the sanitized trusted-HTML path. A
 * hand-crafted URL for a non-public task 404s: getPublicResult re-applies the
 * completed+`useful` predicate.
 *
 * Comment surface is editor-gated even though the page is public: signed-in
 * editors get the full highlight-to-comment flow writing to the same task
 * comments.jsonl as /tasks, while anonymous visitors get a read-only rail
 * (existing comments visible, no composer). `readOnly={!editorAuthed}` drives
 * that split; in read-only mode TaskCommentBody passes `public` to MarkdownDoc
 * so the selection popover + Ask-Claude stay disabled.
 *
 * The body source carries its own `# <title>` H1 (clean-result spec), which
 * MarkdownDoc renders and the TOC picks up as the first entry — so the page
 * header stays a compact breadcrumb + meta strip, no duplicate title.
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { getComments } from "@/lib/tasks";
import { getPublicResult, type ResultConfidence } from "@/lib/results";
import {
  TaskCommentBody,
  type TaskCommentView,
} from "@/app/tasks/[id]/TaskCommentBody";

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

  // Comment flow. The route is publicly viewable, but the comment WRITE
  // surface is editor-gated: signed-in editors get the full anchored-comment
  // flow (writes land in the SAME task comments.jsonl the /tasks page uses),
  // while anonymous visitors get a read-only rail (existing comments + marks
  // visible, no composer, a "Sign in to comment" prompt). `readOnly` is the
  // inverse of editor-auth; TaskCommentBody passes `public` to MarkdownDoc
  // when readOnly OR not editor-authed, which disables the selection popover.
  const editorAuthed = await isEditorAuthed();
  const user = await requireSessionAuth();
  const initialComments: TaskCommentView[] = getComments(id)
    .filter(
      (c) => c.kind === "anchor-comment" || c.kind === "anchor-comment-reply",
    )
    .map((c) => ({
      id: c.id,
      ts: c.ts,
      author: c.author,
      kind: c.kind as "anchor-comment" | "anchor-comment-reply",
      body: c.body,
      anchor: readAnchor(c),
      in_reply_to: c.in_reply_to,
      archived: (c as Record<string, unknown>).archived === true,
    }));

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

      <TaskCommentBody
        taskId={id}
        body={result.body}
        title={result.title}
        isLegacyHtml={result.isLegacyHtml}
        initialComments={initialComments}
        editorAuthed={editorAuthed}
        currentUserEmail={user?.email ?? null}
        readOnly={!editorAuthed}
      />
    </article>
  );
}

// Pull the nested anchor (`{quote, prefix?, suffix?}`) off a raw comment row.
// Task comment rows store the anchor nested (NOT a top-level `quote`).
function readAnchor(
  c: Record<string, unknown>,
): { quote: string; prefix?: string; suffix?: string } | undefined {
  const a = c.anchor;
  if (!a || typeof a !== "object") return undefined;
  const quote = (a as { quote?: unknown }).quote;
  if (typeof quote !== "string" || !quote.trim()) return undefined;
  const out: { quote: string; prefix?: string; suffix?: string } = { quote };
  const prefix = (a as { prefix?: unknown }).prefix;
  const suffix = (a as { suffix?: unknown }).suffix;
  if (typeof prefix === "string") out.prefix = prefix;
  if (typeof suffix === "string") out.suffix = suffix;
  return out;
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
