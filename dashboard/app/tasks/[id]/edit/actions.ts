"use server";

/**
 * Server actions for the task body editor.
 *
 * - saveTaskBody: writes new body via `uv run python scripts/task.py
 *   set-body <N> --file <tmpfile>`. The CLI acquires a flock on
 *   ~/.task-workflow/lock + commits one git commit per call, so concurrent
 *   /issue sessions cannot corrupt the file. We do NOT pass --snapshot —
 *   snapshots are reserved for analyzer's clean-result promotion.
 *
 * - verifyTaskBody: runs `uv run python scripts/verify_task_body.py
 *   --body-stdin` against the in-editor draft (without writing). Returns
 *   the verifier's stdout so the Editor can render PASS/FAIL inline.
 */
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { revalidatePath } from "next/cache";
import { REPO_ROOT } from "@/lib/repo";
import { isEditorAuthed } from "@/lib/auth";

const execFileP = promisify(execFile);

const MAX_BODY_BYTES = 1_000_000; // ~1 MB upper bound; bodies are typically <100 KB

/**
 * Resolve `uv` to an absolute path. The systemd unit (`eps-dashboard.service`)
 * runs with a minimal PATH that does NOT include `~/.local/bin`, so a bare
 * `execFile("uv", ...)` returns `spawn uv ENOENT`. Look in the common
 * install locations, falling back to "uv" so a richer dev PATH still works.
 */
function resolveUv(): string {
  const candidates = [
    process.env.UV_BIN,
    path.join(homedir(), ".local", "bin", "uv"),
    "/usr/local/bin/uv",
    "/usr/bin/uv",
  ].filter((p): p is string => typeof p === "string" && p.length > 0);
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return "uv";
}

type ActionResult = { ok: true } | { ok: false; error: string };

function validateTaskId(raw: unknown): number | null {
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 1) return null;
  return n;
}

export async function saveTaskBody(taskId: number, body: string): Promise<ActionResult> {
  if (!(await isEditorAuthed())) return { ok: false, error: "unauthorized" };
  const id = validateTaskId(taskId);
  if (id === null) return { ok: false, error: "invalid task id" };
  if (typeof body !== "string") return { ok: false, error: "body must be a string" };
  if (body.length > MAX_BODY_BYTES) {
    return { ok: false, error: `body exceeds ${MAX_BODY_BYTES} bytes` };
  }
  const tmp = await mkdtemp(path.join(tmpdir(), "eps-edit-"));
  const tmpPath = path.join(tmp, `body-${id}.md`);
  await writeFile(tmpPath, body, "utf8");
  try {
    await execFileP(
      resolveUv(),
      ["run", "python", "scripts/task.py", "set-body", String(id), "--file", tmpPath],
      { cwd: REPO_ROOT, timeout: 30_000 },
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: `task.py set-body failed: ${msg}` };
  }
  revalidatePath(`/tasks/${id}`);
  revalidatePath(`/tasks/${id}/edit`);
  revalidatePath("/");
  return { ok: true };
}

export async function verifyTaskBody(body: string): Promise<{ ok: boolean; output: string }> {
  if (!(await isEditorAuthed())) {
    return { ok: false, output: "unauthorized" };
  }
  if (typeof body !== "string") {
    return { ok: false, output: "body must be a string" };
  }
  if (body.length > MAX_BODY_BYTES) {
    return { ok: false, output: `body exceeds ${MAX_BODY_BYTES} bytes` };
  }
  return new Promise((resolve) => {
    const child = execFile(
      resolveUv(),
      ["run", "python", "scripts/verify_task_body.py", "--body-stdin"],
      { cwd: REPO_ROOT, timeout: 30_000 },
      (err, stdout, stderr) => {
        const output = `${stdout}${stderr ? `\n${stderr}` : ""}`;
        resolve({ ok: !err, output });
      },
    );
    child.stdin?.write(body);
    child.stdin?.end();
  });
}
