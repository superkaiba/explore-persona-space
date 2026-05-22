import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { getComments, getEvents, getTask, type TaskEvent } from "@/lib/tasks";
import { STATUS_LABELS, type Status } from "@/lib/repo";

export const dynamic = "force-dynamic";

export default async function TaskDetail({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) notFound();
  const task = getTask(id);
  if (!task) notFound();
  const events = getEvents(id);
  const comments = getComments(id);
  const status = task.status;

  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href="/" className="hover:text-stone-800">
            ← All tasks
          </Link>
          <span>·</span>
          <span className="font-mono">#{id}</span>
          <StatusPill status={status} />
        </div>
        <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
          {task.frontmatter.title || "(untitled)"}
        </h1>
        <FrontmatterBar fm={task.frontmatter} />
      </header>

      <section className="prose prose-stone max-w-none sm:prose-lg">
        {task.isLegacyHtml ? (
          <div
            className="legacy-sagan-card"
            // Legacy Sagan-card bodies are trusted HTML authored by our analyzer
            // for our own consumption. Rendered as-is.
            dangerouslySetInnerHTML={{ __html: task.body }}
          />
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw, rehypeHighlight]}
            components={{
              // The page header already renders the task title as an <h1>,
              // and the clean-result spec requires a duplicate `# <title>`
              // line in body source. Suppress that body-level H1 so the
              // title only appears once.
              h1: () => null,
              // `## Figure` is a structural label required by
              // verify_task_body.py but it adds no signal to the rendered
              // view — the image and its caption speak for themselves.
              // Drop the literal "Figure" heading; preserve other H2s.
              h2: ({ children, ...rest }) => {
                const text = Array.isArray(children)
                  ? children.join("")
                  : String(children ?? "");
                if (text.trim() === "Figure") return null;
                return <h2 {...rest}>{children}</h2>;
              },
            }}
          >
            {task.body}
          </ReactMarkdown>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-base font-semibold tracking-tight text-stone-900">
          Timeline · {events.length} event{events.length === 1 ? "" : "s"}
        </h2>
        <Timeline events={events} />
      </section>

      <section>
        <h2 className="mb-3 text-base font-semibold tracking-tight text-stone-900">
          Comments · {comments.length}
        </h2>
        {comments.length === 0 ? (
          <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-6 text-center text-sm text-stone-500">
            No comments yet. (Auth + comment composer land in step 5.)
          </p>
        ) : (
          <ul className="space-y-2">
            {comments.map((c) => (
              <li
                key={c.id}
                className="rounded border border-stone-200 bg-white p-3 text-sm"
              >
                <div className="mb-1 flex items-center gap-2 text-xs text-stone-500">
                  <span className="font-medium text-stone-700">{c.author}</span>
                  <span>·</span>
                  <span>{c.kind}</span>
                  <span>·</span>
                  <time>{c.ts}</time>
                </div>
                <div className="whitespace-pre-wrap">{c.body}</div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </article>
  );
}

function FrontmatterBar({ fm }: { fm: Record<string, unknown> }) {
  const chips: { label: string; value: string }[] = [];
  if (fm.kind) chips.push({ label: "kind", value: String(fm.kind) });
  if (fm.classification) chips.push({ label: "classification", value: String(fm.classification) });
  if (fm.parent_id) chips.push({ label: "parent", value: `#${fm.parent_id}` });
  if (fm.pod_name) chips.push({ label: "pod", value: String(fm.pod_name) });
  if (fm.happy_session_id) chips.push({ label: "session", value: String(fm.happy_session_id) });
  if (fm.has_clean_result) chips.push({ label: "clean-result", value: "true" });
  const tags = Array.isArray(fm.tags) ? (fm.tags as string[]) : [];
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {chips.map((c) => (
        <span
          key={c.label}
          className="rounded border border-stone-200 bg-white px-2 py-0.5"
        >
          <span className="text-stone-500">{c.label}:</span>{" "}
          <span className="font-medium text-stone-800">
            {c.label === "parent" ? (
              <Link
                href={`/tasks/${String(c.value).replace(/^#/, "")}`}
                className="hover:underline"
              >
                {c.value}
              </Link>
            ) : (
              c.value
            )}
          </span>
        </span>
      ))}
      {tags.map((t) => (
        <span
          key={`tag-${t}`}
          className="rounded bg-stone-100 px-2 py-0.5 text-stone-700"
        >
          #{t}
        </span>
      ))}
    </div>
  );
}

function StatusPill({ status }: { status: Status }) {
  return (
    <span className="rounded bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-700">
      {STATUS_LABELS[status]}
    </span>
  );
}

function Timeline({ events }: { events: TaskEvent[] }) {
  if (events.length === 0) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-6 text-center text-sm text-stone-500">
        No events recorded.
      </p>
    );
  }
  // Reverse so most-recent is first
  const sorted = [...events].reverse();
  return (
    <ol className="space-y-1.5">
      {sorted.map((ev, idx) => (
        <li
          key={idx}
          className="rounded border border-stone-200 bg-white px-3 py-2 text-sm"
        >
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-xs text-stone-500">
            <code className="font-mono text-stone-700">{ev.kind}</code>
            <time className="tabular-nums">{ev.ts}</time>
            {ev.by && <span>· {ev.by}</span>}
            {ev.from && ev.to && (
              <span>
                · {ev.from} → {ev.to}
              </span>
            )}
          </div>
          {typeof ev.note === "string" && ev.note.trim() && (
            <details className="mt-1">
              <summary className="cursor-pointer text-xs text-stone-500 hover:text-stone-700">
                {ev.note.slice(0, 120)}
                {ev.note.length > 120 ? "…" : ""}
              </summary>
              <pre className="mt-2 max-h-96 overflow-auto rounded bg-stone-50 p-3 text-xs whitespace-pre-wrap">
                {ev.note}
              </pre>
            </details>
          )}
        </li>
      ))}
    </ol>
  );
}
