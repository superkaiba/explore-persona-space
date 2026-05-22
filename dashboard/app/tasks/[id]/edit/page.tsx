import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { getTask } from "@/lib/tasks";
import { getEditorSecret, isEditorAuthed } from "@/lib/auth";
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

  // If editing is disabled at the server level, hide the route entirely.
  if (!getEditorSecret()) notFound();

  // Gate on cookie; if missing, send to sign-in with `?next=` set.
  if (!(await isEditorAuthed())) {
    redirect(`/sign-in?next=${encodeURIComponent(`/tasks/${id}/edit`)}`);
  }

  const task = getTask(id);
  if (!task) notFound();

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
