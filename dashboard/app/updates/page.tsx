/**
 * /updates — the mentor-facing feed of in-flight + recently-promoted
 * clean results, each surfacing a per-card "Ask Claude" affordance plus
 * a global chat overlay (MentorClaudePanel). All chat traffic flows
 * through the local sidecar at 127.0.0.1:7654.
 *
 * Reads `tasks/` directly via @/lib/tasks#recentTasksForUpdates. No DB.
 * Renders server-side; the interactive bits (cards, chat) are client
 * components that hydrate.
 */
import { CleanResultsLogUpdate } from "@/components/updates/CleanResultsUpdate";
import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import type { CleanResult, CleanResultConfidence } from "@/lib/update-results";
import { markdownExcerpt } from "@/lib/update-results";
import { recentTasksForUpdates, type UpdateTaskRow } from "@/lib/tasks";

export const dynamic = "force-dynamic";

function parseConfidence(title: string): CleanResultConfidence {
  const m = title.match(/\((HIGH|MODERATE|LOW)\s+confidence\)\s*$/i);
  if (!m) return null;
  return m[1].toUpperCase() as CleanResultConfidence;
}

function rowToCleanResult(row: UpdateTaskRow): CleanResult {
  const body = row.isLegacyHtml ? "" : row.body;
  const excerpt = body ? markdownExcerpt(body) : row.title;
  return {
    id: `task-${row.id}`,
    title: row.title || `Task #${row.id}`,
    body,
    excerpt,
    confidence: parseConfidence(row.title),
    // "useful" here means "promoted as useful". For tasks still in flight,
    // we set false so the badge reads as the neutral grey "in progress".
    // Once classification is "useful" we surface the green check.
    useful: row.classification === "useful",
    githubIssueNumber: row.id,   // task id == GitHub issue number historically
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
    href: `/tasks/${row.id}`,
  };
}

export default async function UpdatesPage() {
  const rows = recentTasksForUpdates({ limit: 20, recentDays: 14 });
  const results = rows.map(rowToCleanResult);
  const generatedAt = new Date();
  // The proxy middleware already gates /updates behind sign-in, so the
  // session cookie is always present here. We pass the email down so
  // each card knows whether to show the comment composer + the per-
  // comment delete affordance.
  const user = await requireSessionAuth();
  // Editor-cookie gate for the inline WYSIWYG body editor. Dan signs in
  // with SITE_PASSWORD (session cookie) and gets the read-only view +
  // comments. Only the owner with EDITOR_SECRET gets the "Edit" button
  // in the modal full-view. The server route re-checks this on POST.
  const canEdit = await isEditorAuthed();

  return (
    <CleanResultsLogUpdate
      results={results}
      generatedAt={generatedAt}
      showWeeklyLink={false}
      showInternalLink={false}
      description="Recent in-flight experiments and freshly-promoted clean results."
      currentUserEmail={user?.email ?? null}
      canEdit={canEdit}
    />
  );
}
