import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { getTask, isPaperTask } from "@/lib/tasks";
import { getSitePassword, isEditorAuthed } from "@/lib/auth";
import { Editor } from "./Editor";

export const dynamic = "force-dynamic";

export default async function EditTaskBody({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) notFound();

  // If the site password isn't configured, no one can hold an editor
  // session — hide the route entirely.
  if (!getSitePassword()) notFound();

  // Gate on cookie; if missing, send to sign-in with `?next=` set.
  if (!(await isEditorAuthed())) {
    redirect(`/sign-in?next=${encodeURIComponent(`/tasks/${id}/edit`)}`);
  }

  const task = getTask(id);
  if (!task) notFound();

  // Paper-tasks have a thin paper-stub body.md; the canonical clean-result is
  // the LaTeX paper edited in git. Disable the in-app body editor and show a
  // notice instead of a broken editor that would overwrite the stub.
  if (isPaperTask(task.frontmatter)) {
    return (
      <article className="space-y-4">
        <header className="space-y-2">
          <div className="flex items-baseline gap-3 text-sm text-stone-500">
            <Link href={`/tasks/${id}`} className="hover:text-stone-800">
              ← Back to task
            </Link>
            <span>·</span>
            <span className="font-mono">#{id}</span>
          </div>
          <h1 className="text-xl font-semibold leading-snug tracking-tight">
            Paper-task — body editing disabled
          </h1>
        </header>
        <p className="rounded border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          This is a paper-task (<code>paper: true</code>). Its <code>body.md</code>{" "}
          is a thin paper-stub; the canonical clean-result is the LaTeX paper.
          Edit it in git:{" "}
          <code>docs/papers/issue_{id}/issue_{id}.tex</code> (rebuild with{" "}
          <code>scripts/build_paper.py</code>, verify with{" "}
          <code>scripts/verify_paper.py --issue {id}</code>). The in-app editor is
          disabled here to avoid overwriting the stub.
        </p>
      </article>
    );
  }

  return (
    <article className="space-y-4">
      <header className="space-y-2">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href={`/tasks/${id}`} className="hover:text-stone-800">
            ← Back to task
          </Link>
          <span>·</span>
          <span className="font-mono">#{id}</span>
        </div>
        <h1 className="text-xl font-semibold leading-snug tracking-tight">
          Editing body.md — {task.frontmatter.title || "(untitled)"}
        </h1>
        <p className="text-xs text-stone-500">
          Saves shell out to <code>uv run python scripts/task.py set-body {id} --file …</code>{" "}
          (atomic git commit, flock-protected). Frontmatter (title, status, tags,
          classification) is preserved automatically — those fields have dedicated CLI
          commands (<code>set-title</code>, <code>add-tag</code>, <code>set-clean-result</code>).
          No <code>--snapshot</code> — that&apos;s reserved for analyzer&apos;s clean-result
          promotion.
        </p>
      </header>
      <Editor taskId={id} initialBody={task.body} />
    </article>
  );
}
