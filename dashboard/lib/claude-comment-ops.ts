/**
 * Shared helpers used by `/api/updates/comment` (Change C: @claude
 * dispatch) and `/api/updates/address-comments` (Change D: bulk address).
 *
 * Stays server-only — imports `node:child_process`, `node:fs`, and calls
 * `process.env`. Do not import from `"use client"` files.
 *
 * Pieces:
 *   - `classifyIntent`: Haiku call returning "answer" | "body-edit" |
 *     "code-edit". Defaults to "answer" on any error.
 *   - `streamSidecarChat`: SSE pump for the local sidecar `/chat`. Used
 *     by the `answer` and `body-edit` paths.
 *   - `runClaudeCodeEdit`: spawns the local `claude` CLI inside
 *     `dashboard/`, guards the diff against a forbidden-path list,
 *     commits/builds/restarts on success, reverts on failure. Serialized
 *     by an in-process mutex so two simultaneous code-edits never race
 *     the working tree.
 *   - `parseAddressJson`: stack-based JSON tail extraction so the
 *     address-comments path can split the "revised body" prose from the
 *     trailing `{"addressed": {...}}` blob.
 */
import { spawn } from "node:child_process";
import { promises as fs, existsSync } from "node:fs";
import path from "node:path";
import { homedir } from "node:os";

import { mintSidecarToken } from "@/lib/sidecar-token";
import { REPO_ROOT } from "@/lib/repo";

export type ClassifiedIntent = "answer" | "body-edit" | "code-edit";

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
const HAIKU_MODEL = "claude-haiku-4-5-20251001";

const CLASSIFIER_SYSTEM = [
  'You classify a user comment as one of: "answer", "body-edit", "code-edit".',
  "",
  '- "body-edit": user wants the CONTENT of a research result modified',
  "  (typos, rewording, adding/removing sentences, formatting fixes within",
  "  the experiment write-up).",
  '- "code-edit": user wants the DASHBOARD INTERFACE itself modified',
  "  (colors, layout, padding, new UI elements, bugs in the page rendering).",
  '- "answer": user is asking a question or making a remark that does NOT',
  "  require any edit.",
  "",
  'Return strictly JSON: {"intent": "answer" | "body-edit" | "code-edit"}.',
].join("\n");

/**
 * Classify the mentor's `@claude` comment into one of three intents.
 * Returns "answer" on any failure (parse error, HTTP, missing key,
 * timeout) — the safest default since it just spawns the existing
 * conversational reply path.
 */
export async function classifyIntent(commentBody: string): Promise<ClassifiedIntent> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return "answer";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8_000);
  try {
    const res = await fetch(ANTHROPIC_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: HAIKU_MODEL,
        max_tokens: 32,
        system: CLASSIFIER_SYSTEM,
        messages: [
          { role: "user", content: commentBody.slice(0, 4_000) },
        ],
      }),
      signal: controller.signal,
    });
    if (!res.ok) return "answer";
    const json = (await res.json()) as {
      content?: Array<{ type?: string; text?: string }>;
    };
    const text = (json.content ?? [])
      .filter((b) => b.type === "text")
      .map((b) => b.text ?? "")
      .join("");
    // The model may wrap JSON in prose; extract the first balanced {...}.
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) return "answer";
    try {
      const parsed = JSON.parse(text.slice(start, end + 1)) as { intent?: unknown };
      const v = parsed.intent;
      if (v === "body-edit" || v === "code-edit") return v;
      return "answer";
    } catch {
      return "answer";
    }
  } catch {
    return "answer";
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Stream the local sidecar `/chat` and assemble the assistant text.
 * Returns `null` if the sidecar is unconfigured or the upstream fails.
 *
 * We pump SSE event-by-event and concatenate `token` events. Tool-use
 * events are ignored: the sidecar may emit them but we don't surface
 * intermediate tool calls.
 */
export async function streamSidecarChat({
  sessionId,
  prompt,
  timeoutMs = 5 * 60 * 1000,
  maxChars = 40_000,
}: {
  sessionId: string;
  prompt: string;
  timeoutMs?: number;
  maxChars?: number;
}): Promise<string | null> {
  const tokenResult = await mintSidecarToken();
  if (!tokenResult.ok) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let upstream: Response;
  try {
    upstream = await fetch(`${tokenResult.baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${tokenResult.token}`,
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        session_id: sessionId,
        provider: "claude_code",
        messages: [{ role: "user", content: prompt }],
      }),
      signal: controller.signal,
    });
  } catch {
    clearTimeout(timer);
    return null;
  }
  if (!upstream.ok || !upstream.body) {
    clearTimeout(timer);
    return null;
  }
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let assembled = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const chunks = buf.split(/\r?\n\r?\n/);
      buf = chunks.pop() ?? "";
      for (const eventText of chunks) {
        const parsed = parseSseEventServer(eventText);
        if (!parsed) continue;
        if (parsed.eventName === "token") {
          const t = parsed.data.text;
          if (typeof t === "string") assembled += t;
        }
        // `done` and `error` are swallowed; partial text still returns.
      }
    }
  } finally {
    clearTimeout(timer);
    try {
      reader.releaseLock();
    } catch {
      // detached on timeout
    }
  }
  const text = assembled.trim();
  if (!text) return null;
  return text.length > maxChars ? text.slice(0, maxChars) : text;
}

function parseSseEventServer(
  eventText: string,
): { eventName: string; data: Record<string, unknown> } | null {
  if (!eventText.trim()) return null;
  let eventName = "message";
  let dataStr = "";
  for (const line of eventText.split(/\r?\n/)) {
    if (line.startsWith("event: ")) eventName = line.slice(7).trim();
    if (line.startsWith("data: ")) dataStr += line.slice(6).trim();
  }
  if (!dataStr) return null;
  try {
    return { eventName, data: JSON.parse(dataStr) as Record<string, unknown> };
  } catch {
    return null;
  }
}

/* -------------------------------------------------------------------------- *
 * code-edit dispatch — spawn the local `claude` CLI inside dashboard/,
 * guard the diff, commit/build/restart on success, revert on failure.
 *
 * Forbidden-path globs (anything matching here aborts + reverts):
 * -------------------------------------------------------------------------- */

// Paths checked relative to repo root (the form `git diff --name-only`
// emits). Anything matching here triggers a hard abort. Outside
// `dashboard/` is also forbidden — see `isOutsideDashboard`.
const FORBIDDEN_REGEXES: RegExp[] = [
  /^dashboard\/lib\/auth\.ts$/,
  /^dashboard\/lib\/rate-limit\.ts$/,
  /^dashboard\/lib\/sidecar-token\.ts$/,
  /^dashboard\/app\/api\/auth\//,
  /^dashboard\/app\/api\/sidecar\//,
  /^dashboard\/\.env/,
];

function isForbiddenPath(p: string): boolean {
  if (!p.startsWith("dashboard/")) return true; // out of sandbox
  for (const re of FORBIDDEN_REGEXES) if (re.test(p)) return true;
  return false;
}

function resolveBin(envVar: string, candidates: string[]): string | null {
  const fromEnv = process.env[envVar];
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  for (const c of candidates) if (existsSync(c)) return c;
  return null;
}

function resolveClaudeBin(): string | null {
  return resolveBin("CLAUDE_BIN", [
    path.join(homedir(), ".local", "bin", "claude"),
    "/usr/local/bin/claude",
  ]);
}

function resolveGitBin(): string {
  return "/usr/bin/git";
}

function execShell(
  cmd: string,
  args: string[],
  opts: { cwd?: string; timeoutMs?: number; env?: NodeJS.ProcessEnv } = {},
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd,
      env: opts.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let killed = false;
    const t = opts.timeoutMs
      ? setTimeout(() => {
          killed = true;
          try {
            child.kill("SIGKILL");
          } catch {
            // already dead
          }
        }, opts.timeoutMs)
      : null;
    child.stdout?.on("data", (d) => {
      stdout += String(d);
    });
    child.stderr?.on("data", (d) => {
      stderr += String(d);
    });
    child.on("close", (code) => {
      if (t) clearTimeout(t);
      resolve({
        code,
        stdout,
        stderr: killed ? stderr + "\n[killed: timeout]" : stderr,
      });
    });
    child.on("error", (err) => {
      if (t) clearTimeout(t);
      resolve({ code: -1, stdout, stderr: stderr + `\n[spawn error: ${err.message}]` });
    });
  });
}

let codeEditLock: Promise<unknown> = Promise.resolve();

export type CodeEditResult =
  | { ok: true; sha: string; summary: string; tail: string }
  | { ok: false; error: string; tail: string };

/**
 * Run a guarded `claude` CLI edit inside `dashboard/`. Always serialized
 * via `codeEditLock` so two simultaneous comments cannot race the
 * working tree.
 *
 * Steps:
 *   1. Auto-stash any pre-existing dashboard/ changes (tracked +
 *      untracked) so the edit lands on a clean base. The stash is popped
 *      in `finally`; if pop conflicts, the stash entry is preserved and
 *      surfaced in the result so the user can recover with `git stash list`.
 *   2. Spawn `claude -p <prompt>` with cwd=dashboard/, bypassPermissions.
 *   3. `git diff --name-only` (including staged): if empty → "no-op"; if
 *      any path is forbidden or escapes dashboard/ → revert + abort.
 *   4. `git add dashboard/` + commit.
 *   5. `npm run build` in dashboard/. Capture tail.
 *   6. Build OK: `sudo systemctl restart eps-dashboard.service`.
 *      Build FAIL: `git reset --hard HEAD~1` (the commit) + restart.
 */
export async function runClaudeCodeEdit({
  userComment,
  commentId,
}: {
  userComment: string;
  commentId: string;
}): Promise<CodeEditResult> {
  if (process.env.EPS_CLAUDE_COMMENT_EDIT_DISABLED === "1") {
    return {
      ok: false,
      error: "Code edits via comment are disabled (EPS_CLAUDE_COMMENT_EDIT_DISABLED=1).",
      tail: "",
    };
  }
  // Cap prompt at 8000 chars to bound prompt size.
  const promptText = userComment.slice(0, 8_000);
  const sanitizedCommentId = commentId.replace(/[^a-z0-9-]/gi, "").slice(0, 64) || "unknown";

  // Serialize.
  let release: () => void = () => {};
  const myTurn = new Promise<void>((r) => {
    release = r;
  });
  const prev = codeEditLock;
  codeEditLock = prev.then(() => myTurn);
  await prev;
  try {
    return await _runClaudeCodeEditInner({ promptText, sanitizedCommentId });
  } finally {
    release();
  }
}

async function _runClaudeCodeEditInner({
  promptText,
  sanitizedCommentId,
}: {
  promptText: string;
  sanitizedCommentId: string;
}): Promise<CodeEditResult> {
  const git = resolveGitBin();
  const claudeBin = resolveClaudeBin();
  if (!claudeBin) {
    return {
      ok: false,
      error: "claude CLI not found (set CLAUDE_BIN or install at ~/.local/bin/claude).",
      tail: "",
    };
  }
  const dashboardDir = path.join(REPO_ROOT, "dashboard");

  // Step 1 — auto-stash any pre-existing dashboard/ changes so the edit
  // lands on a clean base. Includes untracked files. The stash is popped
  // in the `finally` below regardless of outcome; if the pop conflicts
  // (e.g. claude touched a file that was also stashed), we leave the
  // stash entry and surface its name so the user can recover.
  let stashRef: string | null = null;
  {
    const r = await execShell(
      git,
      ["-C", REPO_ROOT, "status", "--porcelain", "--", "dashboard"],
      { timeoutMs: 30_000 },
    );
    if (r.code !== 0) {
      return {
        ok: false,
        error: `git status failed (exit ${r.code}).`,
        tail: r.stderr.slice(-4_000),
      };
    }
    if (r.stdout.trim().length > 0) {
      const stashMsg = `eps-dashboard auto-stash before code-edit ${sanitizedCommentId} ${new Date().toISOString()}`;
      const s = await execShell(
        git,
        [
          "-C",
          REPO_ROOT,
          "stash",
          "push",
          "--include-untracked",
          "-m",
          stashMsg,
          "--",
          "dashboard",
        ],
        { timeoutMs: 30_000 },
      );
      if (s.code !== 0) {
        return {
          ok: false,
          error: `auto-stash of pre-existing dashboard/ changes failed (exit ${s.code}).`,
          tail: (s.stderr || s.stdout).slice(-4_000),
        };
      }
      stashRef = stashMsg;
    }
  }

  let inner: CodeEditResult = {
    ok: false,
    error: "unexpected: code-edit body did not return a result",
    tail: "",
  };
  try {
    inner = await _runClaudeCodeEditBody({
      promptText,
      sanitizedCommentId,
      dashboardDir,
      git,
      claudeBin,
    });
  } finally {
    if (stashRef) {
      const pop = await execShell(
        git,
        ["-C", REPO_ROOT, "stash", "pop"],
        { timeoutMs: 30_000 },
      );
      if (pop.code !== 0) {
        const note = `\n\nNote: auto-stash pop conflicted. Your pre-existing dashboard/ changes are preserved in the stash entry: "${stashRef}". Recover with: git stash list && git stash apply <ref>`;
        if (inner.ok) {
          inner = {
            ok: true,
            sha: inner.sha,
            summary: inner.summary,
            tail: inner.tail + note,
          };
        } else {
          inner = {
            ok: false,
            error: inner.error,
            tail: inner.tail + note,
          };
        }
      }
    }
  }
  return inner;
}

async function _runClaudeCodeEditBody({
  promptText,
  sanitizedCommentId,
  dashboardDir,
  git,
  claudeBin,
}: {
  promptText: string;
  sanitizedCommentId: string;
  dashboardDir: string;
  git: string;
  claudeBin: string;
}): Promise<CodeEditResult> {

  // Step 2 — spawn claude. Use `-p` non-interactive mode with the
  // user's prompt passed verbatim. `--permission-mode bypassPermissions`
  // mirrors the sidecar's behavior — required to let Claude edit files
  // without an interactive approval round-trip.
  const claudePrompt = [
    "You are editing the EPS dashboard codebase (Next.js, TypeScript).",
    "Cwd is the dashboard/ directory. You may edit files under dashboard/{app,components,lib}/.",
    "Do NOT edit dashboard/lib/auth.ts, dashboard/lib/rate-limit.ts, dashboard/lib/sidecar-token.ts,",
    "dashboard/app/api/auth/, dashboard/app/api/sidecar/, or any dashboard/.env file.",
    "Do NOT touch anything outside dashboard/.",
    "When finished, do not commit or build — those happen externally.",
    "",
    "User request:",
    promptText,
  ].join("\n");
  const claudeRes = await execShell(
    claudeBin,
    ["-p", claudePrompt, "--permission-mode", "bypassPermissions"],
    { cwd: dashboardDir, timeoutMs: 10 * 60 * 1000 },
  );
  if (claudeRes.code !== 0) {
    // Belt-and-suspenders: revert any partial dashboard/ changes.
    await execShell(git, ["-C", REPO_ROOT, "checkout", "--", "dashboard"], { timeoutMs: 30_000 });
    return {
      ok: false,
      error: `claude CLI exited ${claudeRes.code}.`,
      tail:
        (claudeRes.stdout + "\n" + claudeRes.stderr).split("\n").slice(-30).join("\n"),
    };
  }

  // Step 3 — diff check.
  const diff = await execShell(
    git,
    ["-C", REPO_ROOT, "diff", "--name-only", "HEAD", "--"],
    { timeoutMs: 30_000 },
  );
  const changedPaths = diff.stdout
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
  if (changedPaths.length === 0) {
    return {
      ok: false,
      error: "claude made no file changes.",
      tail: claudeRes.stdout.split("\n").slice(-30).join("\n"),
    };
  }
  const forbidden = changedPaths.filter((p) => isForbiddenPath(p));
  if (forbidden.length > 0) {
    await execShell(git, ["-C", REPO_ROOT, "checkout", "--", "."], { timeoutMs: 30_000 });
    await execShell(git, ["-C", REPO_ROOT, "clean", "-fd", "dashboard/"], { timeoutMs: 30_000 });
    return {
      ok: false,
      error: `claude touched forbidden paths: ${forbidden.join(", ")}. Reverted.`,
      tail: changedPaths.join("\n"),
    };
  }

  // Step 4 — stage + commit.
  {
    const r = await execShell(
      git,
      ["-C", REPO_ROOT, "add", "dashboard/"],
      { timeoutMs: 30_000 },
    );
    if (r.code !== 0) {
      await execShell(git, ["-C", REPO_ROOT, "checkout", "--", "dashboard"], { timeoutMs: 30_000 });
      return { ok: false, error: `git add failed`, tail: r.stderr.slice(-4_000) };
    }
  }
  {
    const msg = `dashboard: claude code-edit via comment ${sanitizedCommentId}`;
    const r = await execShell(
      git,
      [
        "-C",
        REPO_ROOT,
        "-c",
        "user.name=Claude (comment-edit)",
        "-c",
        "user.email=claude@updates",
        "commit",
        "-m",
        msg,
      ],
      { timeoutMs: 60_000 },
    );
    if (r.code !== 0) {
      // Unstage + revert the worktree.
      await execShell(git, ["-C", REPO_ROOT, "reset", "HEAD"], { timeoutMs: 30_000 });
      await execShell(git, ["-C", REPO_ROOT, "checkout", "--", "dashboard"], { timeoutMs: 30_000 });
      return { ok: false, error: `git commit failed`, tail: r.stderr.slice(-4_000) };
    }
  }
  let sha = "unknown";
  {
    const r = await execShell(
      git,
      ["-C", REPO_ROOT, "log", "-1", "--format=%h"],
      { timeoutMs: 10_000 },
    );
    if (r.code === 0) sha = r.stdout.trim();
  }

  // Step 5 — build.
  const build = await execShell(
    "/usr/bin/npm",
    ["run", "build"],
    { cwd: dashboardDir, timeoutMs: 5 * 60 * 1000 },
  );
  const buildTail = (build.stdout + "\n" + build.stderr).split("\n").slice(-30).join("\n");
  if (build.code !== 0) {
    // Revert the commit and restart for safety.
    await execShell(
      git,
      ["-C", REPO_ROOT, "reset", "--hard", "HEAD~1"],
      { timeoutMs: 30_000 },
    );
    await execShell(
      "/usr/bin/sudo",
      ["-n", "/usr/bin/systemctl", "restart", "eps-dashboard.service"],
      { timeoutMs: 60_000 },
    );
    return {
      ok: false,
      error: `npm run build failed (exit ${build.code}). Reverted commit ${sha}.`,
      tail: buildTail,
    };
  }

  // Step 6 — restart service.
  const restart = await execShell(
    "/usr/bin/sudo",
    ["-n", "/usr/bin/systemctl", "restart", "eps-dashboard.service"],
    { timeoutMs: 60_000 },
  );
  if (restart.code !== 0) {
    return {
      ok: false,
      error: `service restart failed (exit ${restart.code}). Commit ${sha} stands.`,
      tail: (restart.stderr || restart.stdout).slice(-2_000),
    };
  }
  return {
    ok: true,
    sha,
    summary: `Edited ${changedPaths.length} file${changedPaths.length === 1 ? "" : "s"}: ${changedPaths.slice(0, 6).join(", ")}${changedPaths.length > 6 ? ", …" : ""}`,
    tail: buildTail,
  };
}

/* -------------------------------------------------------------------------- *
 * body-edit helper.
 *
 * `writeTaskBodyUnchecked` shells out to `task.py set-body` directly (no
 * `isEditorAuthed()` re-check) because the calling route has already
 * verified editor auth at the request boundary. The Server Action
 * `saveTaskBody` reads cookies via `next/headers`, which doesn't survive
 * fire-and-forget detachment — so we need this no-cookie path for the
 * `void spawnClaudeReply(...)` branch in the comment route.
 * -------------------------------------------------------------------------- */

export async function writeTaskBodyUnchecked(
  taskId: number,
  body: string,
): Promise<{ ok: true } | { ok: false; error: string }> {
  if (!Number.isFinite(taskId) || !Number.isInteger(taskId) || taskId < 1) {
    return { ok: false, error: "invalid task id" };
  }
  if (typeof body !== "string") return { ok: false, error: "body must be a string" };
  if (body.length > 1_000_000) return { ok: false, error: "body exceeds 1MB" };
  const tmpDir = await fs.mkdtemp(path.join((await import("node:os")).tmpdir(), "eps-claude-"));
  const tmpFile = path.join(tmpDir, `body-${taskId}.md`);
  await fs.writeFile(tmpFile, body, "utf8");
  const uv = resolveBin("UV_BIN", [
    path.join(homedir(), ".local", "bin", "uv"),
    "/usr/local/bin/uv",
    "/usr/bin/uv",
  ]) ?? "uv";
  const r = await execShell(
    uv,
    ["run", "python", "scripts/task.py", "set-body", String(taskId), "--file", tmpFile],
    { cwd: REPO_ROOT, timeoutMs: 30_000 },
  );
  if (r.code !== 0) {
    return { ok: false, error: `task.py set-body failed: ${(r.stderr || r.stdout).slice(-2_000)}` };
  }
  return { ok: true };
}

/**
 * Set a task's `track` frontmatter field via the CLI mutator. Unlike body
 * frontmatter (which `set-body` preserves + strips-from-new content), `track`
 * MUST go through `task.py set-track`, which mutates the frontmatter dict
 * under flock + one git commit + registry sync.
 */
export async function writeTaskTrackUnchecked(
  taskId: number,
  track: "experiment" | "human",
): Promise<{ ok: true } | { ok: false; error: string }> {
  if (!Number.isFinite(taskId) || !Number.isInteger(taskId) || taskId < 1) {
    return { ok: false, error: "invalid task id" };
  }
  if (track !== "experiment" && track !== "human") {
    return { ok: false, error: "invalid track" };
  }
  const uv =
    resolveBin("UV_BIN", [
      path.join(homedir(), ".local", "bin", "uv"),
      "/usr/local/bin/uv",
      "/usr/bin/uv",
    ]) ?? "uv";
  const r = await execShell(
    uv,
    ["run", "python", "scripts/task.py", "set-track", String(taskId), track],
    { cwd: REPO_ROOT, timeoutMs: 30_000 },
  );
  if (r.code !== 0) {
    return {
      ok: false,
      error: `task.py set-track failed: ${(r.stderr || r.stdout).slice(-2_000)}`,
    };
  }
  return { ok: true };
}

export async function readHeadSha(): Promise<string> {
  const r = await execShell(
    resolveGitBin(),
    ["-C", REPO_ROOT, "log", "-1", "--format=%h"],
    { timeoutMs: 10_000 },
  );
  return r.code === 0 ? r.stdout.trim() : "unknown";
}

export function buildBodyEditPrompt({
  currentBody,
  userComment,
  taskId,
}: {
  currentBody: string;
  userComment: string;
  taskId: number;
}): string {
  return [
    `You are editing the body markdown of EPS experiment write-up task #${taskId}.`,
    "The user requested the change below. Return the FULL revised markdown body, NOTHING ELSE.",
    "No commentary, no preamble, no code fences around the whole thing — just the markdown verbatim.",
    "",
    "User request:",
    userComment,
    "",
    "Current body:",
    currentBody,
  ].join("\n");
}

/* -------------------------------------------------------------------------- *
 * Address-comments JSON tail extractor.
 *
 * The model returns: <revised body>\n\n{"addressed": {<id>: <note>, ...}}
 *
 * We need to split the prose from the JSON tail. Simple regex breaks
 * when the body itself contains `{`s. Scan from the end with a brace
 * stack and split on the matching open-brace.
 * -------------------------------------------------------------------------- */

export function parseAddressJson(
  fullResponse: string,
): { body: string; addressed: Record<string, string> } | null {
  // Find the last balanced `{...}` group in the string.
  const s = fullResponse.trimEnd();
  const lastClose = s.lastIndexOf("}");
  if (lastClose === -1) return null;
  let depth = 0;
  let openIdx = -1;
  for (let i = lastClose; i >= 0; i--) {
    const ch = s[i];
    if (ch === "}") depth++;
    else if (ch === "{") {
      depth--;
      if (depth === 0) {
        openIdx = i;
        break;
      }
    }
  }
  if (openIdx === -1) return null;
  const jsonBlob = s.slice(openIdx, lastClose + 1);
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonBlob);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const addressed = (parsed as { addressed?: unknown }).addressed;
  if (!addressed || typeof addressed !== "object") return null;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(addressed as Record<string, unknown>)) {
    out[k] = typeof v === "string" ? v : String(v);
  }
  const body = s.slice(0, openIdx).trimEnd();
  if (!body) return null;
  return { body, addressed: out };
}
