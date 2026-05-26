/**
 * /updates/[id] — standalone full-page view of ONE result card.
 *
 * Same content as the modal overlay on /updates, but a real route so
 * the Fullscreen button can `window.open` it in a new browser window.
 * Owner sees Edit + Address comments buttons; non-owners see read +
 * comment surface only. CardCommentBox (rail layout) renders the body
 * with TOC, anchored comments, composer.
 *
 * Reads `tasks/<status>/<N>/body.md` via @/lib/tasks. No DB.
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { CardCommentBox } from "@/components/updates/CardCommentBox";
import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import {
  recentTasksForUpdates,
  type UpdateTaskRow,
} from "@/lib/tasks";
import { markdownExcerpt } from "@/lib/update-results";

export const dynamic = "force-dynamic";

function findTask(id: number): UpdateTaskRow | null {
  // Use the same recent-tasks pull the /updates page uses; if missing,
  // widen the window. Keeps a single source of truth (no extra DB call).
  const wide = recentTasksForUpdates({ limit: 500, recentDays: 3650 });
  return wide.find((r) => r.id === id) ?? null;
}

export default async function StandaloneResultPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: rawId } = await params;
  const taskId = Number(rawId);
  if (!Number.isFinite(taskId) || !Number.isInteger(taskId) || taskId < 1) {
    notFound();
  }
  const row = findTask(taskId);
  if (!row) notFound();

  const user = await requireSessionAuth();
  const canEdit = await isEditorAuthed();
  const body = row.isLegacyHtml ? "" : row.body;

  return (
    <article className="min-h-dvh bg-canvas">
      <header className="sticky top-0 z-10 border-b border-border bg-panel px-6 py-4 shadow-sm">
        <div className="mx-auto flex max-w-7xl items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-muted">
              <Link
                href="/updates"
                className="inline-flex items-center gap-1 rounded border border-border bg-subtle px-1.5 py-0.5 font-mono text-fg hover:bg-raised"
              >
                <ArrowLeft className="h-3 w-3" />
                Back to updates
              </Link>
              <span className="rounded border border-border bg-subtle px-1.5 py-0.5 font-mono text-fg">
                #{row.id}
              </span>
              <span className="font-mono">{row.status}</span>
              <span className="font-mono">{row.classification}</span>
            </div>
            <h1 className="text-xl font-semibold leading-snug text-fg">
              {row.title || `Task #${row.id}`}
            </h1>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-8">
        {body ? (
          <CardCommentBox
            taskId={row.id}
            body={body}
            currentUserEmail={user?.email ?? null}
            layout="rail"
          />
        ) : (
          <p className="rounded border border-dashed border-border bg-subtle px-4 py-6 text-sm text-muted">
            This task has no body content yet
            {row.isLegacyHtml ? " (legacy HTML body)" : ""}.
          </p>
        )}
        {!canEdit && (
          <p className="mt-8 rounded border border-dashed border-border bg-subtle px-3 py-2 text-[11px] text-muted">
            You&apos;re signed in as a viewer. Sign in with the editor secret
            on <Link href="/sign-in" className="text-accent hover:underline">/sign-in</Link>{" "}
            to enable Edit + Address-comments.
          </p>
        )}
      </div>
    </article>
  );
}

void markdownExcerpt; // silence unused
