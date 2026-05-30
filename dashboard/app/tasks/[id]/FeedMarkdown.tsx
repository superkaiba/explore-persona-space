"use client";

/**
 * FeedMarkdown — the client markdown surface for EVERY card in the task feed
 * (plan card, event cards). Replaces the plain `EventNoteMarkdown` so plan +
 * event notes become commentable through the same <MarkdownDoc> keystone the
 * body card uses.
 *
 * Why a thin wrapper rather than dropping <MarkdownDoc> straight into the
 * server-component cards: `onCommentCreate` is a function (POST + refetch) and
 * functions can't cross the RSC server→client prop boundary. So this client
 * component pulls `onCommentCreate` off <AnchoredCommentsContext> (set once by
 * the page-level <TaskFeed> provider) and hands it to <MarkdownDoc>. When no
 * provider/hook is mounted above (e.g. a public surface), it falls back to a
 * read-only render — `onCommentCreate` is `null`, so MarkdownDoc keeps the
 * legacy non-composer behavior.
 *
 * Perf: this renders ONLY when its enclosing <CollapsiblePanel> is expanded
 * (plan + event cards are `defaultCollapsed`, and CollapsiblePanel doesn't
 * mount children while collapsed). The markdown parse therefore stays lazy +
 * client-side — same perf posture as the EventNoteMarkdown it replaces.
 *
 * TOC + collapsible-sections are OFF here: the surrounding CollapsiblePanel
 * already owns collapse, and per-note TOCs would be noise.
 */
import { MarkdownDoc } from "@/components/MarkdownDoc";
import { useAnchoredComments } from "@/app/tasks/[id]/AnchoredCommentsContext";

export function FeedMarkdown({
  body,
  docId,
}: {
  body: string;
  /** Namespaces heading ids / collapse state per card so cards don't collide. */
  docId: string;
}) {
  const { onCommentCreate } = useAnchoredComments();
  return (
    <MarkdownDoc
      body={body}
      docId={docId}
      showToc={false}
      enableCollapsibleSections={false}
      onCommentCreate={onCommentCreate ?? undefined}
      public={!onCommentCreate}
    />
  );
}
