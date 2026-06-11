"use client";

/**
 * Invisible visit recorder for the task detail page: marks the task "seen"
 * in localStorage on mount, so the board's unseen-update glow clears even
 * for direct-URL visits that never went through a board card click.
 */
import { useEffect } from "react";
import { markTaskSeen } from "@/components/tasks/task-seen";

export function MarkTaskSeen({ taskId }: { taskId: number }) {
  useEffect(() => {
    markTaskSeen(taskId);
  }, [taskId]);
  return null;
}
