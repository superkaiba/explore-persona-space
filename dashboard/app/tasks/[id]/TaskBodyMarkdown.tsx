"use client";

/**
 * TaskBodyMarkdown — the body-card markdown surface for /tasks/[id].
 *
 * Renders the task body through the shared <MarkdownDoc> keystone, pulling the
 * inline-composer create hook off the PAGE-LEVEL <AnchoredCommentsContext>
 * (set by <TaskFeed>) rather than mounting its own provider. This is the task-
 * page analogue of <TaskCommentBody> (which keeps its own provider for the
 * /results surface). The body keeps TOC + per-header collapse + Ask-Claude.
 *
 * Highlight-to-comment uses the inline composer (anchored at the selection),
 * so there's no buried side rail here; the comment list lives in the page-level
 * <TaskCommentsPanel> above the feed.
 */
import { MarkdownDoc } from "@/components/MarkdownDoc";
import { useAnchoredComments } from "@/app/tasks/[id]/AnchoredCommentsContext";

export function TaskBodyMarkdown({
  taskId,
  body,
  title,
  isLegacyHtml,
  canWrite,
}: {
  taskId: number;
  body: string;
  title: string;
  isLegacyHtml: boolean;
  /** Editor-authed AND not read-only. Gates the composer + Ask-Claude. */
  canWrite: boolean;
}) {
  const { onCommentCreate } = useAnchoredComments();
  return (
    <MarkdownDoc
      body={body}
      isLegacyHtml={isLegacyHtml}
      showToc
      enableCollapsibleSections
      docId={taskId}
      enableAskClaude={canWrite}
      askClaudeTitle={title}
      public={!canWrite}
      onCommentCreate={canWrite ? (onCommentCreate ?? undefined) : undefined}
    />
  );
}
