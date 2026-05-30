/**
 * POST /api/tasks/track — set a task's `track` field.
 *
 *   POST { taskId, track: "experiment" | "human" }  -> { ok, track }
 *
 * Reads the task's `body.md`, sets/replaces the `track:` line in its YAML
 * frontmatter (preserving everything else verbatim), and writes it back
 * through `writeTaskBodyUnchecked` — which shells out to
 * `uv run python scripts/task.py set-body <N> --file <tmp>`, so the CLI
 * acquires flock on ~/.task-workflow/lock + commits one git commit per
 * call. Concurrent dashboard edits cannot corrupt the file.
 *
 * Auth: editor-gated (`requireSessionAuth`), same single-tier site-password
 * gate the body/title editors use. Writes serialize through an in-process
 * per-file mutex (same pattern as the comment route) so two simultaneous
 * track toggles can't race the read-modify-write.
 *
 * The track is stored as plain frontmatter, NOT a status-enum mutation —
 * status stays workflow-owned. CLI support for *creating* human-track
 * tasks (a `--track` flag / new kinds) is a separate follow-up; this route
 * only mutates existing tasks' track in place.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import { requireSessionAuth } from "@/lib/auth";
import { resolveTaskPath } from "@/lib/tasks";
import { writeTaskBodyUnchecked } from "@/lib/claude-comment-ops";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VALID_TRACKS = new Set(["experiment", "human"]);

/* In-process mutex per body.md path — mirrors the comment route. */
const locks = new Map<string, Promise<void>>();

async function withFileLock<T>(file: string, fn: () => Promise<T>): Promise<T> {
  const prev = locks.get(file) ?? Promise.resolve();
  let release: () => void;
  const next = new Promise<void>((resolve) => {
    release = resolve;
  });
  locks.set(
    file,
    prev.then(() => next),
  );
  await prev;
  try {
    return await fn();
  } finally {
    release!();
    if (locks.get(file) === next) locks.delete(file);
  }
}

function validateTaskId(raw: unknown): number | null {
  const n = Number(raw);
  return Number.isFinite(n) && Number.isInteger(n) && n >= 1 ? n : null;
}

/**
 * Set/replace the `track:` line inside the leading YAML frontmatter block,
 * leaving the rest of the document byte-for-byte intact. The body MUST
 * already have a frontmatter block (`---\n...\n---`) — tasks always do, so
 * a missing block is an error (we don't silently invent one).
 *
 * Returns the new full document, or null if there's no frontmatter block.
 */
function setTrackInFrontmatter(raw: string, track: string): string | null {
  // Frontmatter must open at the very start of the file.
  if (!raw.startsWith("---")) return null;
  // Find the closing fence. The opening `---` is line 1; the closing fence
  // is the next line that is exactly `---` (allowing trailing CR).
  const lines = raw.split("\n");
  // lines[0] is the opening `---` (possibly with trailing CR).
  if (lines[0].replace(/\r$/, "").trim() !== "---") return null;
  let closeIdx = -1;
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].replace(/\r$/, "").trim() === "---") {
      closeIdx = i;
      break;
    }
  }
  if (closeIdx === -1) return null;

  // Frontmatter body is lines[1 .. closeIdx-1]. Replace an existing
  // top-level `track:` line if present; else insert one just before the
  // closing fence.
  const trackRe = /^track:\s*.*$/;
  let replaced = false;
  for (let i = 1; i < closeIdx; i++) {
    if (trackRe.test(lines[i].replace(/\r$/, ""))) {
      lines[i] = `track: ${track}`;
      replaced = true;
      break;
    }
  }
  if (!replaced) {
    lines.splice(closeIdx, 0, `track: ${track}`);
  }
  return lines.join("\n");
}

export async function POST(request: Request) {
  const user = await requireSessionAuth();
  if (!user) return Response.json({ ok: false, error: "unauthorized" }, { status: 401 });

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ ok: false, error: "invalid json" }, { status: 400 });
  }
  const obj = (payload ?? {}) as Record<string, unknown>;

  const taskId = validateTaskId(obj.taskId);
  if (taskId === null) {
    return Response.json({ ok: false, error: "invalid taskId" }, { status: 400 });
  }
  const track = String(obj.track ?? "").trim();
  if (!VALID_TRACKS.has(track)) {
    return Response.json(
      { ok: false, error: "track must be 'experiment' or 'human'" },
      { status: 400 },
    );
  }

  const dir = resolveTaskPath(taskId);
  if (!dir) return Response.json({ ok: false, error: "task not found" }, { status: 404 });
  const bodyPath = path.join(dir, "body.md");

  try {
    const result = await withFileLock(bodyPath, async () => {
      let raw: string;
      try {
        raw = await fs.readFile(bodyPath, "utf8");
      } catch {
        return { ok: false as const, status: 404, error: "body.md not found" };
      }
      const next = setTrackInFrontmatter(raw, track);
      if (next === null) {
        return {
          ok: false as const,
          status: 422,
          error: "task body has no YAML frontmatter block",
        };
      }
      if (next === raw) {
        // Already at the requested track — no write needed.
        return { ok: true as const };
      }
      const write = await writeTaskBodyUnchecked(taskId, next);
      if (!write.ok) {
        return { ok: false as const, status: 500, error: write.error };
      }
      return { ok: true as const };
    });

    if (!result.ok) {
      return Response.json({ ok: false, error: result.error }, { status: result.status });
    }
    return Response.json({ ok: true, track });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return Response.json({ ok: false, error: `set-track failed: ${msg}` }, { status: 500 });
  }
}
