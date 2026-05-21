import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { getPlan, getTask } from "@/lib/tasks";
import { STATUS_LABELS, type Status } from "@/lib/repo";

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
        <section className="prose prose-sm sm:prose-base prose-stone max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw, rehypeHighlight]}
          >
            {plan.body}
          </ReactMarkdown>
        </section>
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

function StatusPill({ status }: { status: Status }) {
  return (
    <span className="rounded bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-700">
      {STATUS_LABELS[status]}
    </span>
  );
}
