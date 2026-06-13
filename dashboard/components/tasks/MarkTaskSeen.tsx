"use client";

/**
 * Invisible visit recorder for the task detail page: marks the task "seen"
 * in localStorage on mount, so the board's unseen-update glow clears even
 * for direct-URL visits that never went through a board card click.
 *
 * `serverNow` is the server's render timestamp — the same clock that stamps
 * the lastActivityAt mtimes the glow compares against — so a client clock
 * running behind the VM can't leave the task glowing after this visit.
 */
import { useEffect } from "react";
import { markTaskSeen } from "@/components/tasks/task-seen";

export function MarkTaskSeen({
  taskId,
  serverNow,
}: {
  taskId: number;
  serverNow?: string;
}) {
  useEffect(() => {
    markTaskSeen(taskId, serverNow);
  }, [taskId, serverNow]);
  return null;
}
