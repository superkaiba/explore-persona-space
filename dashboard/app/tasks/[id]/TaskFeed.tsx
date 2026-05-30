"use client";

/**
 * TaskFeed — the ONE page-level anchored-comments shell for /tasks/[id].
 *
 * Replaces the per-card provider that used to live inside the body's
 * <TaskCommentBody>. A SINGLE <AnchoredCommentsProvider> now wraps the WHOLE
 * feed (body + plan + event cards), carrying:
 *   - `anchors`: every open, non-archived `anchor-comment` whose quote is ≥4
 *     chars. Because anchors are quote-based, a comment's <mark> lights up on
 *     EVERY card whose rendered text contains that quote.
 *   - `onCommentCreate`: POST `/api/updates/comment` (taskId) then refetch.
 *     Every <MarkdownDoc> descendant (body via <TaskBodyMarkdown>, plan +
 *     events via <FeedMarkdown>) pulls this off context to drive its inline
 *     composer — so highlight-to-comment works on every card at any width.
 *
 * The comment LIST + delete + hover/scroll lives in <TaskCommentsPanel>, a
 * page-level collapsible panel rendered ABOVE the feed (NOT nested inside the
 * body card's CollapsiblePanel, which is where it used to get buried ~22k px
 * down). Inline creation is the primary path; the panel is for viewing /
 * managing existing comments.
 *
 * The server component (`page.tsx`) builds the feed structure (cards, TOC,
 * "Activity" divider) and passes it in as `children`; this client shell only
 * owns the comment state + provider + panel. Functions can't cross the RSC
 * boundary, hence the context hand-off.
 */
import { useCallback, useMemo, useState } from "react";
import {
  AnchoredCommentsProvider,
  type AnchorRecord,
} from "@/app/tasks/[id]/AnchoredCommentsContext";
import type { TaskCommentView } from "@/app/tasks/[id]/TaskCommentBody";
import { TaskCommentsPanel } from "@/app/tasks/[id]/TaskCommentsPanel";

export function TaskFeed({
  taskId,
  initialComments,
  canWrite,
  currentUserEmail,
  children,
}: {
  taskId: number;
  initialComments: TaskCommentView[];
  /** Editor-authed AND not read-only. Gates writes + the composer hook. */
  canWrite: boolean;
  currentUserEmail: string | null;
  children: React.ReactNode;
}) {
  const [comments, setComments] = useState<TaskCommentView[]>(initialComments);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`/api/updates/comment?taskId=${taskId}`, {
        cache: "no-store",
        credentials: "same-origin",
      });
      if (res.status === 401) return; // unauth viewer — keep server-passed list
      const data = (await res.json()) as
        | { ok: true; comments: TaskCommentView[] }
        | { ok: false; error: string };
      if (data.ok) setComments(data.comments);
    } catch {
      /* leave existing list; transient fetch error */
    }
  }, [taskId]);

  // Archived closure: an archived anchor-comment hides its whole subtree
  // (synthesis reply + any user follow-ups). Iterative; 64-hop sanity cap.
  // Mirrors the old TaskCommentBody.
  const archivedIds = useMemo(() => {
    const closed = new Set<string>();
    for (const c of comments) {
      if (c.kind === "anchor-comment" && c.archived === true) closed.add(c.id);
    }
    let changed = true;
    let hops = 0;
    while (changed && hops < 64) {
      changed = false;
      for (const c of comments) {
        if (c.in_reply_to && closed.has(c.in_reply_to) && !closed.has(c.id)) {
          closed.add(c.id);
          changed = true;
        }
      }
      hops++;
    }
    return closed;
  }, [comments]);

  const visibleComments = useMemo(
    () => comments.filter((c) => !archivedIds.has(c.id)),
    [comments, archivedIds],
  );

  // Committed anchors: open root anchor-comments with a quote ≥4 chars.
  const anchors: AnchorRecord[] = useMemo(
    () =>
      visibleComments
        .filter(
          (c) =>
            c.kind === "anchor-comment" &&
            typeof c.anchor?.quote === "string" &&
            c.anchor.quote.trim().length >= 4,
        )
        .map((c) => ({ id: c.id, quote: (c.anchor!.quote as string).trim() })),
    [visibleComments],
  );

  // The inline-composer create hook shared with every MarkdownDoc on the page.
  // POST the comment, refetch on success, return true so the composer clears.
  // Mirrors the @claude poll cadence the old rail used.
  const onCommentCreate = useCallback(
    async ({ quote, body }: { quote: string; body: string }): Promise<boolean> => {
      if (!canWrite) return false;
      const text = body.trim();
      if (!text) return false;
      try {
        const payload: { taskId: number; body: string; anchor?: { quote: string } } = {
          taskId,
          body: text,
        };
        const q = quote.trim();
        if (q) payload.anchor = { quote: q };
        const res = await fetch("/api/updates/comment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          body: JSON.stringify(payload),
        });
        const data = (await res.json()) as
          | { ok: true; will_reply?: boolean }
          | { ok: false; error: string };
        if (!data.ok) return false;
        await refresh();
        if (data.will_reply) {
          for (const delayMs of [3_000, 8_000, 20_000, 45_000]) {
            setTimeout(() => void refresh(), delayMs);
          }
        }
        return true;
      } catch {
        return false;
      }
    },
    [taskId, canWrite, refresh],
  );

  return (
    <AnchoredCommentsProvider
      anchors={anchors}
      onCommentCreate={canWrite ? onCommentCreate : null}
    >
      <TaskCommentsPanel
        taskId={taskId}
        comments={visibleComments}
        canWrite={canWrite}
        currentUserEmail={currentUserEmail}
        onChanged={refresh}
      />
      {children}
    </AnchoredCommentsProvider>
  );
}
