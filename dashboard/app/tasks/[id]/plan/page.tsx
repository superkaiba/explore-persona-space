import Link from "next/link";
import { notFound } from "next/navigation";
import { getComments, getPlan, getTask } from "@/lib/tasks";
import { STATUS_LABELS, type Status } from "@/lib/repo";
import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { type TaskCommentView } from "@/app/tasks/[id]/TaskCommentBody";
import { PlanCommentBody } from "./PlanCommentBody";

export const dynamic = "force-dynamic";

export default async function TaskPlan({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) notFound();
  const task = getTask(id);
  if (!task) notFound();
  const plan = getPlan(id);

  const canWrite = await isEditorAuthed();
  const user = await requireSessionAuth();
  // Same comment store as /tasks/[id] — anchors posted on the plan view land
  // in tasks/<id>/comments.jsonl and surface on both views.
  const initialComments: TaskCommentView[] = getComments(id)
    .filter((c) => c.kind === "anchor-comment" || c.kind === "anchor-comment-reply")
    .map((c) => ({
      id: c.id,
      ts: c.ts,
      author: c.author,
      kind: c.kind as "anchor-comment" | "anchor-comment-reply",
      body: c.body,
      anchor: readAnchor(c),
      in_reply_to: c.in_reply_to,
      archived: (c as Record<string, unknown>).archived === true,
    }));

  return (
    <article className="space-y-6">
      <header className="space-y-3">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href="/" className="hover:text-stone-800">
            ← All tasks
          </Link>
          <span>·</span>
          <Link href={`/tasks/${id}`} className="hover:text-stone-800">
            ← Task #{id}
          </Link>
          <span>·</span>
          <span className="font-mono">plan</span>
          <StatusPill status={task.status} />
          {plan && (
            <>
              <span>·</span>
              <span className="font-mono text-stone-400">{plan.filename}</span>
            </>
          )}
        </div>
        <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
          Plan · {task.frontmatter.title || `Task #${id}`}
        </h1>
      </header>

      {plan ? (
        <PlanCommentBody
          taskId={id}
          body={plan.body}
          title={typeof task.frontmatter.title === "string" ? task.frontmatter.title : `Task #${id}`}
          initialComments={initialComments}
          canWrite={canWrite}
          currentUserEmail={user?.email ?? null}
        />
      ) : (
        <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-10 text-center text-sm text-stone-500">
          No plan posted yet for task #{id}. Plans are written to{" "}
          <code className="font-mono">tasks/&lt;status&gt;/{id}/plans/v&lt;K&gt;.md</code>{" "}
          by the adversarial-planner via{" "}
          <code className="font-mono">task.py new-plan-version</code>.
        </p>
      )}
    </article>
  );
}

// Pull the nested anchor (`{quote, prefix?, suffix?}`) off a raw comment row.
function readAnchor(
  c: Record<string, unknown>,
): { quote: string; prefix?: string; suffix?: string } | undefined {
  const a = c.anchor;
  if (!a || typeof a !== "object") return undefined;
  const quote = (a as { quote?: unknown }).quote;
  if (typeof quote !== "string" || !quote.trim()) return undefined;
  const out: { quote: string; prefix?: string; suffix?: string } = { quote };
  const prefix = (a as { prefix?: unknown }).prefix;
  const suffix = (a as { suffix?: unknown }).suffix;
  if (typeof prefix === "string") out.prefix = prefix;
  if (typeof suffix === "string") out.suffix = suffix;
  return out;
}

function StatusPill({ status }: { status: Status }) {
  return (
    <span className="rounded bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-700">
      {STATUS_LABELS[status]}
    </span>
  );
}
