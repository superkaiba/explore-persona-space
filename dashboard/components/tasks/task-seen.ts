/**
 * Per-device "have I looked at this task since it last changed?" tracking.
 *
 * Storage is localStorage (single-user dashboard, per-browser is fine):
 *   - `eps:task-seen:v1`          → { [taskId]: ISO ts of the last visit }
 *   - `eps:task-seen-baseline:v1` → ISO ts of the FIRST board visit on this
 *     device. Tasks with no per-id entry fall back to the baseline, so a
 *     fresh browser doesn't light up all ~600 cards at once — glow only
 *     accrues from updates that land after the first visit.
 *
 * A task counts as "unseen" when its server-computed `lastActivityAt`
 * (body.md mtime ⊔ status-entry ts, see lib/tasks.ts) is newer than the
 * device's seen ts for that id (or the baseline).
 *
 * Client-only module: callers are client components ("use client" lives on
 * them); every function no-ops without `window`.
 */

const SEEN_KEY = "eps:task-seen:v1";
const BASELINE_KEY = "eps:task-seen-baseline:v1";

export type SeenState = {
  baseline: string | null;
  seen: Record<string, string>;
};

/** Read the seen map, initializing the first-visit baseline on first call.
 * Corrupt/missing storage degrades to "nothing unseen" (null baseline). */
export function readSeenState(): SeenState {
  if (typeof window === "undefined") return { baseline: null, seen: {} };
  let baseline: string | null = null;
  let seen: Record<string, string> = {};
  try {
    baseline = window.localStorage.getItem(BASELINE_KEY);
    if (!baseline) {
      baseline = new Date().toISOString();
      window.localStorage.setItem(BASELINE_KEY, baseline);
    }
    const raw = window.localStorage.getItem(SEEN_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        seen = parsed as Record<string, string>;
      }
    }
  } catch {
    // localStorage unavailable (private mode / quota) or corrupt JSON —
    // glow simply stays off on this device rather than crashing the board.
    return { baseline: null, seen: {} };
  }
  return { baseline, seen };
}

/**
 * Record "user looked at task <id> now". `atLeastIso` (the task's
 * server-side lastActivityAt, or the server's render time) guards against
 * a client clock running behind the VM: the stamp is max(client now,
 * atLeastIso) so a just-updated task never keeps glowing after a visit.
 */
export function markTaskSeen(id: number, atLeastIso?: string | null): void {
  if (typeof window === "undefined") return;
  try {
    const { seen } = readSeenState();
    let stamp = new Date().toISOString();
    if (atLeastIso) {
      const atLeastMs = Date.parse(atLeastIso);
      if (Number.isFinite(atLeastMs) && atLeastMs > Date.parse(stamp)) {
        stamp = new Date(atLeastMs).toISOString();
      }
    }
    seen[String(id)] = stamp;
    window.localStorage.setItem(SEEN_KEY, JSON.stringify(seen));
  } catch {
    // Same degradation as readSeenState — tracking is best-effort.
  }
}

/**
 * Reset everything to "seen as of now": baseline moves to now, per-id map
 * clears (O(1) storage). Escape hatch for mass-glow events, e.g. a bulk git
 * operation rewriting many body.md mtimes at once.
 */
export function markAllTasksSeen(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(BASELINE_KEY, new Date().toISOString());
    window.localStorage.setItem(SEEN_KEY, "{}");
  } catch {
    // Best-effort, as above.
  }
}
