"use server";

/**
 * Server actions for posting comments on a task.
 *
 * - addComment: appends a comment via `task.py add-comment` (atomic git commit).
 * - askClaude: posts a question comment AND spawns a Claude Code session that
 *   will search the codebase/eval_results and post an answer comment back.
 *   Uses `scripts/spawn_session.py spawn-issue --initial-prompt …` which
 *   routes through the local Happy daemon (HTTP RPC at 127.0.0.1) and starts
 *   a `claude` (Claude Code) subprocess in bypassPermissions mode so it can
 *   run shell commands without a human at the keyboard.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { revalidatePath } from "next/cache";
import { REPO_ROOT } from "@/lib/repo";
import { isEditorAuthed } from "@/lib/auth";

const execFileP = promisify(execFile);

const MAX_COMMENT_BYTES = 50_000;

const COMMENT_KINDS = ["question", "answer", "followup-proposal", "note"] as const;
type CommentKind = (typeof COMMENT_KINDS)[number];

type ActionResult =
  | { ok: true; commentId: string; spawnedSessionId?: string }
  | { ok: false; error: string };

function validateTaskId(raw: unknown): number | null {
  const n = Number(raw);
  return Number.isFinite(n) && Number.isInteger(n) && n >= 1 ? n : null;
}

function validateKind(raw: unknown): CommentKind | null {
  return typeof raw === "string" && (COMMENT_KINDS as readonly string[]).includes(raw)
    ? (raw as CommentKind)
    : null;
}

async function runAddComment(opts: {
  taskId: number;
  author: string;
  kind: CommentKind;
  body: string;
  inReplyTo?: string;
  anchorQuote?: string;
}): Promise<{ commentId: string } | { error: string }> {
  const args = [
    "run",
    "python",
    "scripts/task.py",
    "add-comment",
    String(opts.taskId),
    "--author",
    opts.author,
    "--kind",
    opts.kind,
    "--body",
    "-", // read from stdin
  ];
  if (opts.inReplyTo) {
    args.push("--in-reply-to", opts.inReplyTo);
  }
  if (opts.anchorQuote) {
    args.push("--anchor-quote", opts.anchorQuote);
  }
  return new Promise((resolve) => {
    const child = execFile(
      "uv",
      args,
      { cwd: REPO_ROOT, timeout: 30_000 },
      (err, stdout, stderr) => {
        if (err) {
          resolve({ error: `task.py add-comment failed: ${err.message}\n${stderr}` });
          return;
        }
        const cid = stdout.trim().split(/\s+/).pop() ?? "";
        if (!/^c\d{3,}$/.test(cid)) {
          resolve({ error: `unexpected task.py output: ${JSON.stringify(stdout)}` });
          return;
        }
        resolve({ commentId: cid });
      },
    );
    child.stdin?.write(opts.body);
    child.stdin?.end();
  });
}

export async function addComment(formData: FormData): Promise<ActionResult> {
  if (!(await isEditorAuthed())) return { ok: false, error: "unauthorized" };
  const taskId = validateTaskId(formData.get("taskId"));
  if (taskId === null) return { ok: false, error: "invalid task id" };
  const kind = validateKind(formData.get("kind"));
  if (kind === null) return { ok: false, error: "invalid kind" };
  const body = String(formData.get("body") || "").trim();
  if (!body) return { ok: false, error: "body is empty" };
  if (body.length > MAX_COMMENT_BYTES) {
    return { ok: false, error: `body exceeds ${MAX_COMMENT_BYTES} chars` };
  }
  const anchorQuote = (() => {
    const raw = formData.get("anchorQuote");
    if (typeof raw !== "string") return undefined;
    const q = raw.trim();
    if (!q) return undefined;
    if (q.length > 2_000) return q.slice(0, 2_000);
    return q;
  })();

  const result = await runAddComment({
    taskId,
    author: "thomas",
    kind,
    body,
    anchorQuote,
  });
  if ("error" in result) return { ok: false, error: result.error };
  revalidatePath(`/tasks/${taskId}`);
  return { ok: true, commentId: result.commentId };
}

const ASK_CLAUDE_PROMPT = (taskId: number, questionId: string, question: string) => `\
You are a Claude Code session spawned by the EPS dashboard to answer a
question on task #${taskId}. This is a read-and-answer flow, not an
experiment. Do not run experiments, modify body.md, change task status,
spawn other sessions, or push to remote.

The question (comment id ${questionId}) is:

${question.split("\n").map((l) => `> ${l}`).join("\n")}

Your job:

1. Read the task context:
       uv run python scripts/task.py view ${taskId}
       uv run python scripts/task.py list-comments ${taskId}

2. Investigate. Search the codebase, eval results, and task artifacts:
   - Eval results:  eval_results/issue_${taskId}/
   - Plans:         tasks/*/${taskId}/plans/plan.md
   - Code:          grep/Read under src/, scripts/, configs/, .claude/

3. Post your answer as a markdown comment:
       uv run python scripts/task.py add-comment ${taskId} \\
           --author claude-code \\
           --kind answer \\
           --in-reply-to ${questionId} \\
           --body - < /tmp/answer-${taskId}.md

   Write the answer first to /tmp/answer-${taskId}.md (full markdown, with
   code fences and links to file:line locations), then pipe it in via stdin.

Be concise. If you can't find the answer, say so clearly and list what
you DID check.
`;

export async function askClaude(formData: FormData): Promise<ActionResult> {
  if (!(await isEditorAuthed())) return { ok: false, error: "unauthorized" };
  const taskId = validateTaskId(formData.get("taskId"));
  if (taskId === null) return { ok: false, error: "invalid task id" };
  const question = String(formData.get("body") || "").trim();
  if (!question) return { ok: false, error: "question is empty" };
  if (question.length > MAX_COMMENT_BYTES) {
    return { ok: false, error: `question exceeds ${MAX_COMMENT_BYTES} chars` };
  }

  const anchorQuote = (() => {
    const raw = formData.get("anchorQuote");
    if (typeof raw !== "string") return undefined;
    const q = raw.trim();
    if (!q) return undefined;
    if (q.length > 2_000) return q.slice(0, 2_000);
    return q;
  })();

  // Step 1: post the question as a comment.
  const post = await runAddComment({
    taskId,
    author: "thomas",
    kind: "question",
    body: question,
    anchorQuote,
  });
  if ("error" in post) return { ok: false, error: post.error };
  const questionId = post.commentId;

  // Step 2: spawn a Claude Code session via the Happy daemon RPC.
  // We forward `--initial-prompt` so the session boots with the answer
  // protocol already in place and in bypassPermissions mode.
  const prompt = ASK_CLAUDE_PROMPT(taskId, questionId, question);
  let spawnedSessionId: string | undefined;
  try {
    const { stdout } = await execFileP(
      "uv",
      [
        "run",
        "python",
        "scripts/spawn_session.py",
        "spawn-issue",
        "--issue",
        String(taskId),
        "--initial-prompt",
        prompt,
      ],
      { cwd: REPO_ROOT, timeout: 60_000 },
    );
    const m = stdout.match(/session spawned:\s*(\S+)/);
    if (m) spawnedSessionId = m[1];
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      ok: false,
      error: `question posted as ${questionId} but spawn failed: ${msg}`,
    };
  }

  revalidatePath(`/tasks/${taskId}`);
  return { ok: true, commentId: questionId, spawnedSessionId };
}
