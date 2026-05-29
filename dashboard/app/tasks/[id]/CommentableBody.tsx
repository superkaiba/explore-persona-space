"use client";

/**
 * CommentableBody — thin wrapper around the shared <MarkdownDoc> keystone.
 *
 * This used to be the ~616-line implementation of selection-capture,
 * multi-occurrence <mark> anchoring, collapsible H1/H2/H3 sections,
 * hover-sync, scroll-to, and position publishing. That logic now lives in
 * `components/MarkdownDoc.tsx` (generalized so docs / results / overview /
 * updates can reuse it too). CommentableBody is preserved as a back-compat
 * shim so /tasks/[id] and the updates card renderers keep working with their
 * existing props unchanged.
 *
 * Props are identical to the historical surface:
 *   - body, isLegacyHtml: required render inputs.
 *   - enableCollapsibleSections, taskId: opt-in collapse + its persistence key.
 *
 * `taskId` maps to MarkdownDoc's generic `docId` (the localStorage /
 * event-scoping key). Comment anchoring + the AnchoredCommentsContext wiring
 * are unchanged — MarkdownDoc reads the same context, so the surrounding
 * <AnchoredCommentsProvider> + CommentList behavior is untouched.
 */
import { MarkdownDoc } from "@/components/MarkdownDoc";

export function CommentableBody({
  body,
  isLegacyHtml,
  enableCollapsibleSections = false,
  taskId,
}: {
  body: string;
  isLegacyHtml: boolean;
  enableCollapsibleSections?: boolean;
  taskId?: number;
}) {
  return (
    <MarkdownDoc
      body={body}
      isLegacyHtml={isLegacyHtml}
      enableCollapsibleSections={enableCollapsibleSections}
      docId={taskId}
    />
  );
}
