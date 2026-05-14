/**
 * Repo-root + tasks-dir path resolution.
 *
 * Local dev: process.cwd() is dashboard/; ../tasks is the workspace tasks dir.
 * Vercel: project root is dashboard/, and outputFileTracingIncludes pulls
 * ../tasks/** into the runtime. process.cwd() resolves the same way.
 */
import path from "node:path";

export const REPO_ROOT = path.resolve(process.cwd(), "..");
export const TASKS_DIR = path.join(REPO_ROOT, "tasks");
export const REGISTRY_PATH = path.join(TASKS_DIR, "REGISTRY.json");

export const STATUSES = [
  "proposed",
  "planning",
  "plan_pending",
  "approved",
  "running",
  "verifying",
  "interpreting",
  "reviewing",
  "awaiting_promotion",
  "completed",
  "blocked",
  "archived",
] as const;

export type Status = (typeof STATUSES)[number];

// Display ordering for the homepage. Active work first, then awaiting
// promotion, then everything else. Terminal statuses at the bottom.
export const STATUS_DISPLAY_ORDER: Status[] = [
  "running",
  "interpreting",
  "reviewing",
  "verifying",
  "awaiting_promotion",
  "approved",
  "plan_pending",
  "planning",
  "proposed",
  "blocked",
  "completed",
  "archived",
];

export const STATUS_LABELS: Record<Status, string> = {
  proposed: "To do",
  planning: "Planning",
  plan_pending: "Plan awaiting review",
  approved: "Approved",
  running: "Running",
  verifying: "Verifying uploads",
  interpreting: "Interpreting",
  reviewing: "Reviewing",
  awaiting_promotion: "Awaiting promotion",
  completed: "Completed",
  blocked: "Blocked",
  archived: "Archived",
};

// Statuses expanded by default on mobile / desktop home.
export const STATUS_EXPANDED_BY_DEFAULT: Status[] = [
  "running",
  "awaiting_promotion",
  "proposed",
];
