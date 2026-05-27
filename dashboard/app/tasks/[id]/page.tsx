import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { isEditorAuthed } from "@/lib/auth";
import {
  getComments,
  getEvents,
  getPlan,
  getTask,
  type Frontmatter,
  type Task,
  type TaskComment,
  type TaskEvent,
  type TaskPlan,
} from "@/lib/tasks";
import { STATUS_LABELS, type Status } from "@/lib/repo";
import { CollapsiblePanel } from "@/components/CollapsiblePanel";
import {
  TaskTocSidebar,
  type TocEntry,
} from "@/components/tasks/TaskTocSidebar";
import { EditableBody } from "./EditableBody";
import { TitleEditor } from "./TitleEditor";

export const dynamic = "force-dynamic";

// Event kinds rendered as a thin status-transition pill rather than a full card.
const TRANSITION_KINDS = new Set<string>(["epm:status-changed"]);

// Event kinds rendered as a one-line compact row (bookkeeping noise). The note
// is shown truncated; the kind itself usually says everything.
const COMPACT_KINDS = new Set<string>([
  "epm:codex-task-spawned",
  "epm:codex-task-completed",
  "epm:codex-task-failed",
  "epm:goal-updated",
  "epm:auto-defaults",
  "epm:run-launched",
  "epm:progress",
  "epm:plan-approved",
  "epm:original-body",
  "epm:created",
  "epm:abort",
  "epm:stale",
  "epm:pod-terminated",
]);

// Items in the reverse-chronological feed. Each carries a stable
// `itemKey` (used as the localStorage key for collapse state) and
// `anchorId` (the DOM id the TOC sidebar scroll-jumps to).
type FeedItem =
  | { kind: "body"; ts: string; itemKey: string; anchorId: string; task: Task }
  | {
      kind: "plan";
      ts: string;
      itemKey: string;
      anchorId: string;
      plan: TaskPlan;
      version: number;
    }
  | {
      kind: "event-card";
      ts: string;
      itemKey: string;
      anchorId: string;
      event: TaskEvent;
    }
  | {
      kind: "event-compact";
      ts: string;
      itemKey: string;
      anchorId: string;
      event: TaskEvent;
    }
  | {
      kind: "transition";
      ts: string;
      itemKey: string;
      anchorId: string;
      event: TaskEvent;
    };

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
  const plan = getPlan(id);
  const items = buildFeedItems(task, events, plan);
  const canEdit = await isEditorAuthed();

  const tocEntries: TocEntry[] = items.map(itemToTocEntry);

  return (
    <article className="space-y-6">
      <header className="space-y-3">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href="/" className="hover:text-stone-800">
            ← All tasks
          </Link>
          <span>·</span>
          <span className="font-mono">#{id}</span>
          <StatusPill status={task.status} />
        </div>
        <TitleEditor
          taskId={id}
          initialTitle={task.frontmatter.title ?? ""}
          canEdit={canEdit}
        />
        <FrontmatterBar fm={task.frontmatter} />
      </header>

      <div className="grid gap-6 md:grid-cols-[240px_minmax(0,1fr)]">
        <TaskTocSidebar taskId={id} entries={tocEntries} />

        <div className="min-w-0 space-y-3">
          <section className="space-y-3" aria-label="Task timeline">
            {items.length === 0 ? (
              <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-6 text-center text-sm text-stone-500">
                No content yet.
              </p>
            ) : (
              items.map((it) => (
                <FeedRow
                  key={it.itemKey}
                  item={it}
                  taskId={id}
                  canEdit={canEdit}
                />
              ))
            )}
          </section>

          <section>
            <h2 className="mb-3 mt-6 text-base font-semibold tracking-tight text-stone-900">
              Comments · {comments.length}
            </h2>
            <CommentsList comments={comments} />
          </section>
        </div>
      </div>
    </article>
  );
}

// ─── Feed construction ──────────────────────────────────────────────────────

function buildFeedItems(
  task: Task,
  events: TaskEvent[],
  plan: TaskPlan | null,
): FeedItem[] {
  const items: FeedItem[] = [];
  const created = (task.frontmatter.created_at as string | undefined) ??
    "1970-01-01T00:00:00Z";

  // Body card.
  const bodyTs = computeBodyTs(task, events, plan, created);
  items.push({
    kind: "body",
    ts: bodyTs,
    itemKey: "body",
    anchorId: "feed-body",
    task,
  });

  // Plan card (the actual plan document — separate from the epm:plan event).
  if (plan) {
    const planEvents = events.filter((e) => e.kind === "epm:plan");
    const latestPlanEvent = planEvents.length
      ? planEvents[planEvents.length - 1]
      : undefined;
    // Bump the plan-card timestamp 1 ms past the event so they sort adjacent
    // but plan-document-first on the same wall-clock instant.
    const baseTs = latestPlanEvent?.ts ?? created;
    const planTs = bumpMs(baseTs, 1);
    const version = (latestPlanEvent?.version as number | undefined) ?? 1;
    const itemKey = `plan-v${version}`;
    items.push({
      kind: "plan",
      ts: planTs,
      itemKey,
      anchorId: `feed-${itemKey}`,
      plan,
      version,
    });
  }

  // Track repeated (ts, kind) tuples so two events that share both still
  // get unique itemKeys. events.jsonl rarely has true duplicates but the
  // analyzer occasionally posts back-to-back markers within 1 ms.
  const seen = new Map<string, number>();
  for (const ev of events) {
    if (ev.kind === "epm:created") continue; // body covers creation
    const base = `${ev.ts}|${ev.kind}`;
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    const itemKey = `event-${ev.ts}-${ev.kind}${n === 0 ? "" : `-${n}`}`;
    const anchorId = `feed-${slugifyKey(itemKey)}`;
    if (TRANSITION_KINDS.has(ev.kind)) {
      items.push({ kind: "transition", ts: ev.ts, itemKey, anchorId, event: ev });
    } else if (COMPACT_KINDS.has(ev.kind)) {
      items.push({
        kind: "event-compact",
        ts: ev.ts,
        itemKey,
        anchorId,
        event: ev,
      });
    } else {
      items.push({
        kind: "event-card",
        ts: ev.ts,
        itemKey,
        anchorId,
        event: ev,
      });
    }
  }

  // Sort reverse-chronological.
  items.sort((a, b) => (a.ts < b.ts ? 1 : a.ts > b.ts ? -1 : 0));
  return items;
}

// Make the itemKey safe to use as a DOM id (alphanumerics, hyphens, underscores
// only — `:` and `.` are valid in HTML id but break CSS selectors).
function slugifyKey(key: string): string {
  return key.replace(/[^A-Za-z0-9_-]/g, "-");
}

// Body floats to the top when it's the freshest artifact:
//   - has_clean_result=true and promoted_at set  → use promoted_at
//   - has_clean_result=true, no promoted_at      → max(events): the analyzer
//       has replaced body in place but the user hasn't promoted yet
//   - plan exists, no clean-result               → keep at created_at (plan
//       sits above; body is the original spec)
//   - no plan, no clean-result                   → max(events): body is the
//       canonical artifact for proposed/early-stage tasks
function computeBodyTs(
  task: Task,
  events: TaskEvent[],
  plan: TaskPlan | null,
  created: string,
): string {
  if (task.frontmatter.has_clean_result) {
    const promoted = task.frontmatter.promoted_at as string | undefined;
    if (promoted) return promoted;
    return events.reduce((acc, ev) => (ev.ts > acc ? ev.ts : acc), created);
  }
  if (plan) return created;
  return events.reduce((acc, ev) => (ev.ts > acc ? ev.ts : acc), created);
}

function bumpMs(iso: string, ms: number): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  return new Date(t + ms).toISOString();
}

// ─── TOC entries ────────────────────────────────────────────────────────────

function itemToTocEntry(item: FeedItem): TocEntry {
  switch (item.kind) {
    case "body":
      return {
        itemKey: item.itemKey,
        anchorId: item.anchorId,
        label: item.task.frontmatter.has_clean_result
          ? "Clean result"
          : "Original body",
        kind: "body",
        ts: item.ts,
      };
    case "plan":
      return {
        itemKey: item.itemKey,
        anchorId: item.anchorId,
        label: `Plan v${item.version}`,
        kind: "plan",
        ts: item.ts,
      };
    case "event-card":
      return {
        itemKey: item.itemKey,
        anchorId: item.anchorId,
        label: shortEventLabel(item.event),
        kind: "event-card",
        ts: item.ts,
      };
    case "event-compact":
      return {
        itemKey: item.itemKey,
        anchorId: item.anchorId,
        label: shortEventLabel(item.event),
        kind: "event-compact",
        ts: item.ts,
      };
    case "transition": {
      const from = typeof item.event.from === "string" ? item.event.from : "?";
      const to = typeof item.event.to === "string" ? item.event.to : "?";
      return {
        itemKey: item.itemKey,
        anchorId: item.anchorId,
        label: `${from} → ${to}`,
        kind: "transition",
        ts: item.ts,
      };
    }
  }
}

function shortEventLabel(event: TaskEvent): string {
  const note = typeof event.note === "string" ? event.note.trim() : "";
  const firstLine = note.split("\n", 1)[0] ?? "";
  if (firstLine) {
    return firstLine.length > 80 ? firstLine.slice(0, 80) + "…" : firstLine;
  }
  return event.kind;
}

// ─── Row dispatch ───────────────────────────────────────────────────────────

function FeedRow({
  item,
  taskId,
  canEdit,
}: {
  item: FeedItem;
  taskId: number;
  canEdit: boolean;
}) {
  switch (item.kind) {
    case "body":
      return (
        <BodyCard
          task={item.task}
          taskId={taskId}
          itemKey={item.itemKey}
          anchorId={item.anchorId}
          canEdit={canEdit}
        />
      );
    case "plan":
      return (
        <PlanCard
          plan={item.plan}
          version={item.version}
          taskId={taskId}
          itemKey={item.itemKey}
          anchorId={item.anchorId}
        />
      );
    case "event-card":
      return (
        <EventCard
          event={item.event}
          taskId={taskId}
          itemKey={item.itemKey}
          anchorId={item.anchorId}
        />
      );
    case "event-compact":
      return (
        <EventCompactRow
          event={item.event}
          taskId={taskId}
          itemKey={item.itemKey}
          anchorId={item.anchorId}
        />
      );
    case "transition":
      return (
        <TransitionPill
          event={item.event}
          taskId={taskId}
          itemKey={item.itemKey}
          anchorId={item.anchorId}
        />
      );
  }
}

// ─── Cards ──────────────────────────────────────────────────────────────────

function BodyCard({
  task,
  taskId,
  itemKey,
  anchorId,
  canEdit,
}: {
  task: Task;
  taskId: number;
  itemKey: string;
  anchorId: string;
  canEdit: boolean;
}) {
  const isCleanResult = !!task.frontmatter.has_clean_result;
  const label = isCleanResult
    ? "Clean result · task body"
    : "Original task body";
  const renderedBody = task.isLegacyHtml ? (
    <div
      className="prose prose-stone max-w-none sm:prose-lg legacy-sagan-card"
      // Legacy Sagan-card bodies are trusted HTML authored by our analyzer
      // for our own consumption. Rendered as-is.
      dangerouslySetInnerHTML={{ __html: task.body }}
    />
  ) : (
    <div className="prose prose-stone max-w-none sm:prose-lg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeHighlight]}
        components={{
          // The page header already renders the task title as <h1>; the
          // clean-result spec requires a duplicate `# <title>` line in
          // body source. Suppress to avoid double display.
          h1: () => null,
          // `## Figure` is a structural label required by
          // verify_task_body.py but adds no signal to the rendered view.
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
    </div>
  );

  return (
    <CollapsiblePanel
      taskId={taskId}
      itemKey={itemKey}
      anchorId={anchorId}
      header={
        <>
          <span className="font-mono text-stone-700">{label}</span>
          {task.frontmatter.created_at && (
            <time className="tabular-nums">
              {String(task.frontmatter.created_at)}
            </time>
          )}
          {isCleanResult && task.frontmatter.classification && (
            <span>· classification: {String(task.frontmatter.classification)}</span>
          )}
        </>
      }
    >
      {task.isLegacyHtml ? (
        renderedBody
      ) : (
        <EditableBody
          taskId={taskId}
          initialBody={task.body}
          canEdit={canEdit}
        >
          {renderedBody}
        </EditableBody>
      )}
    </CollapsiblePanel>
  );
}

function PlanCard({
  plan,
  version,
  taskId,
  itemKey,
  anchorId,
}: {
  plan: TaskPlan;
  version: number;
  taskId: number;
  itemKey: string;
  anchorId: string;
}) {
  return (
    <CollapsiblePanel
      taskId={taskId}
      itemKey={itemKey}
      anchorId={anchorId}
      emphasis="plan"
      header={
        <>
          <span className="rounded bg-amber-100 px-1.5 py-0.5 font-mono text-[11px] font-medium text-amber-900">
            PLAN v{version}
          </span>
          <span className="font-mono text-stone-500">{plan.filename}</span>
        </>
      }
    >
      {/* Permalink lives in the body, not the header — nesting an <a>
          inside the panel's toggle <button> is invalid HTML and the
          RSC boundary refuses event handlers on the header slot. */}
      <div className="mb-3 flex justify-end">
        <Link
          href={`/tasks/${taskId}/plan`}
          className="text-xs text-stone-500 hover:text-stone-800"
        >
          permalink ↗
        </Link>
      </div>
      <div className="prose prose-stone max-w-none sm:prose-base">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw, rehypeHighlight]}
        >
          {plan.body}
        </ReactMarkdown>
      </div>
    </CollapsiblePanel>
  );
}

function EventCard({
  event,
  taskId,
  itemKey,
  anchorId,
}: {
  event: TaskEvent;
  taskId: number;
  itemKey: string;
  anchorId: string;
}) {
  return (
    <CollapsiblePanel
      taskId={taskId}
      itemKey={itemKey}
      anchorId={anchorId}
      header={
        <>
          <code className="font-mono font-medium text-stone-800">
            {event.kind}
            {typeof event.version === "number" ? ` v${event.version}` : ""}
          </code>
          <time className="tabular-nums">{event.ts}</time>
          {event.by && event.by !== "unknown" && <span>· {event.by}</span>}
        </>
      }
    >
      {typeof event.note === "string" && event.note.trim() ? (
        <div className="prose prose-sm prose-stone max-w-none sm:prose-base">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw, rehypeHighlight]}
          >
            {event.note}
          </ReactMarkdown>
        </div>
      ) : (
        <p className="text-xs text-stone-500">(no note body)</p>
      )}
    </CollapsiblePanel>
  );
}

// Compact one-line events render as their own collapsible row so the TOC
// sidebar can still scroll to them. They start collapsed since the kind
// and note preview live in the header.
function EventCompactRow({
  event,
  taskId,
  itemKey,
  anchorId,
}: {
  event: TaskEvent;
  taskId: number;
  itemKey: string;
  anchorId: string;
}) {
  const note = typeof event.note === "string" ? event.note : "";
  const preview = note.length > 180 ? note.slice(0, 180) + "…" : note;
  return (
    <CollapsiblePanel
      taskId={taskId}
      itemKey={itemKey}
      anchorId={anchorId}
      defaultCollapsed
      header={
        <>
          <code className="font-mono text-stone-700">{event.kind}</code>
          <time className="tabular-nums text-stone-500">{event.ts}</time>
          {event.by && event.by !== "unknown" && (
            <span className="text-stone-500">· {event.by}</span>
          )}
          {preview && (
            <span className="text-stone-600">
              · <span className="whitespace-pre-wrap">{preview}</span>
            </span>
          )}
        </>
      }
    >
      {note ? (
        <pre className="overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-stone-700">
          {note}
        </pre>
      ) : (
        <p className="text-xs text-stone-500">(no note body)</p>
      )}
    </CollapsiblePanel>
  );
}

function TransitionPill({
  event,
  taskId,
  itemKey,
  anchorId,
}: {
  event: TaskEvent;
  taskId: number;
  itemKey: string;
  anchorId: string;
}) {
  const from = typeof event.from === "string" ? event.from : "?";
  const to = typeof event.to === "string" ? event.to : "?";
  // Wrap in the same CollapsiblePanel shell so the TOC entry scrolls to
  // a section element with the matching `id={anchorId}`. We pass
  // `alwaysExpanded` because the pill itself is the entire content —
  // there's nothing to toggle. The header becomes the pill.
  return (
    <CollapsiblePanel
      taskId={taskId}
      itemKey={itemKey}
      anchorId={anchorId}
      alwaysExpanded
      header={
        <div className="flex w-full items-center gap-3 py-0 text-xs text-stone-500">
          <div className="h-px flex-1 bg-stone-200" />
          <span className="rounded bg-stone-100 px-2 py-0.5 font-mono">
            status: {from} → {to}
          </span>
          <time className="tabular-nums">{event.ts}</time>
          <div className="h-px flex-1 bg-stone-200" />
        </div>
      }
    >
      {/* `alwaysExpanded` keeps the body slot mounted but the header
          carries the pill itself; nothing to render here. */}
      <></>
    </CollapsiblePanel>
  );
}

// ─── Shared atoms ───────────────────────────────────────────────────────────

function FrontmatterBar({ fm }: { fm: Frontmatter }) {
  const chips: { label: string; value: string }[] = [];
  if (fm.kind) chips.push({ label: "kind", value: String(fm.kind) });
  if (fm.classification)
    chips.push({ label: "classification", value: String(fm.classification) });
  if (fm.parent_id) chips.push({ label: "parent", value: `#${fm.parent_id}` });
  if (fm.pod_name) chips.push({ label: "pod", value: String(fm.pod_name) });
  if (fm.happy_session_id)
    chips.push({ label: "session", value: String(fm.happy_session_id) });
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

function CommentsList({ comments }: { comments: TaskComment[] }) {
  if (comments.length === 0) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-6 text-center text-sm text-stone-500">
        No comments yet. (Auth + comment composer land in step 5.)
      </p>
    );
  }
  return (
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
  );
}
