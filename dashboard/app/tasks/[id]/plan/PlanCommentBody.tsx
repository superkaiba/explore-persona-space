"use client";

/**
 * PlanCommentBody — the client shell for the standalone plan view
 * (/tasks/[id]/plan).
 *
 * Mounts the SAME page-level anchored-comments machinery the task feed uses,
 * scoped to the same taskId so a comment posted here lands in the same
 * `tasks/<id>/comments.jsonl` (and shows on /tasks/[id] too). The plan body
 * renders through <MarkdownDoc> with TOC + the inline composer; the comment
 * list + whole-task composer live in <TaskCommentsPanel> above the plan.
 *
 * Reuses <TaskFeed> as the provider/panel host and renders the plan body via
 * an inner client component that pulls the create hook off context.
 */
import { MarkdownDoc } from "@/components/MarkdownDoc";
import { useAnchoredComments } from "@/app/tasks/[id]/AnchoredCommentsContext";
import { TaskFeed } from "@/app/tasks/[id]/TaskFeed";
import type { TaskCommentView } from "@/app/tasks/[id]/TaskCommentBody";

export function PlanCommentBody({
  taskId,
  body,
  title,
  initialComments,
  canWrite,
  currentUserEmail,
}: {
  taskId: number;
  body: string;
  title: string;
  initialComments: TaskCommentView[];
  canWrite: boolean;
  currentUserEmail: string | null;
}) {
  return (
    <TaskFeed
      taskId={taskId}
      initialComments={initialComments}
      canWrite={canWrite}
      currentUserEmail={currentUserEmail}
    >
      <div className="mt-4">
        <PlanMarkdown taskId={taskId} body={body} title={title} canWrite={canWrite} />
      </div>
    </TaskFeed>
  );
}

function PlanMarkdown({
  taskId,
  body,
  title,
  canWrite,
}: {
  taskId: number;
  body: string;
  title: string;
  canWrite: boolean;
}) {
  const { onCommentCreate } = useAnchoredComments();
  return (
    <MarkdownDoc
      body={body}
      showToc
      enableCollapsibleSections
      docId={`plan-${taskId}`}
      enableAskClaude={canWrite}
      askClaudeTitle={`Plan · ${title}`}
      public={!canWrite}
      onCommentCreate={canWrite ? (onCommentCreate ?? undefined) : undefined}
    />
  );
}
