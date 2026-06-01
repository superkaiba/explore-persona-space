/**
 * /questions — PUBLIC research hub.
 *
 * Auto-rendered from `docs/open_questions.md` via `lib/questions.ts`. The doc
 * is the single source of truth: the writer-side flow (`/issue` +
 * `scripts/living_docs.py`) maintains the anchors + evidence trailers, and
 * this page parses them at request time (force-dynamic mirrors every other
 * disk-reading route).
 *
 * Grouping is by HEADING LEVEL (not "nearest H3") so apps land under their
 * own H2 region rather than getting swept into §4. The parser handles the
 * level walk; this page just emits one section per group in display order
 * and suppresses empty groups (e.g. Settled today).
 *
 * Public-vs-gated link routing comes from `evidenceHrefForTaskId` so the
 * /questions surface, the /results/[id] reverse-index block, and the
 * overview transform all share ONE source of truth for the `useful`
 * predicate. Non-public ids render as gated `/tasks/<id>` links; no titles
 * or excerpts leak into the public HTML, and the Link prefetch is disabled
 * so an idle hover doesn't trigger a server-side render of the gated route.
 */
import Link from "next/link";
import {
  listQuestions,
  type Question,
  type QuestionConfidence,
} from "@/lib/questions";
import { evidenceHrefForTaskId, publicResultIdSet } from "@/lib/results";
import { QuestionsBrowser } from "./QuestionsBrowser";

export const dynamic = "force-dynamic";

// Stable display order: question sections first (the 4 H3 names in doc
// order), then Applications, then Settled (always last). Groups not in the
// doc are skipped automatically because we filter on the loaded set.
const QUESTION_SECTION_ORDER: string[] = [
  "Distance between contexts",
  "Updating (W, C) toward a behavior — what installs, at what cost?",
  "Generalization — how an update at (C, B) propagates to (C′, B′)",
  "What are contexts and behaviors — the C–B duality",
];

const CONFIDENCE_STYLE: Record<
  QuestionConfidence,
  { chip: string; dot: string }
> = {
  HIGH: { chip: "bg-emerald-50 text-emerald-700 border-emerald-200", dot: "bg-emerald-500" },
  MODERATE: { chip: "bg-amber-50 text-amber-800 border-amber-200", dot: "bg-amber-500" },
  LOW: { chip: "bg-stone-100 text-stone-600 border-stone-200", dot: "bg-stone-400" },
};

type Group = {
  /** Display heading. */
  name: string;
  /** Ordered list of `(subsection | null) -> Question[]` for stable rendering. */
  subgroups: { subsection: string | null; questions: Question[] }[];
  /** Whether this group's rows should expose status badges (questions yes, apps no). */
  kind: "questions" | "applications" | "settled";
};

function groupQuestions(questions: Question[]): Group[] {
  // Bucket by section, preserving doc order within each subsection.
  const bySection = new Map<string, Question[]>();
  for (const q of questions) {
    if (!bySection.has(q.section)) bySection.set(q.section, []);
    bySection.get(q.section)!.push(q);
  }
  const out: Group[] = [];
  // Open question sections in canonical doc order.
  for (const section of QUESTION_SECTION_ORDER) {
    const rows = bySection.get(section);
    if (!rows || rows.length === 0) continue;
    out.push({
      name: section,
      kind: "questions",
      subgroups: bucketBySubsection(rows),
    });
  }
  // Applications.
  const apps = bySection.get("Applications");
  if (apps && apps.length > 0) {
    out.push({
      name: "Applications",
      kind: "applications",
      subgroups: [{ subsection: null, questions: apps }],
    });
  }
  // Settled — always last, suppressed when empty (the doc currently has no
  // settled entries, so this group simply doesn't render).
  const settled = bySection.get("Settled");
  if (settled && settled.length > 0) {
    out.push({
      name: "Settled",
      kind: "settled",
      subgroups: bucketBySubsection(settled),
    });
  }
  return out;
}

function bucketBySubsection(
  rows: Question[],
): { subsection: string | null; questions: Question[] }[] {
  const out: { subsection: string | null; questions: Question[] }[] = [];
  const indexFor = new Map<string, number>();
  for (const q of rows) {
    const key = q.subsection ?? "__none__";
    let i = indexFor.get(key);
    if (i === undefined) {
      i = out.length;
      indexFor.set(key, i);
      out.push({ subsection: q.subsection ?? null, questions: [] });
    }
    out[i].questions.push(q);
  }
  return out;
}

export default async function QuestionsPage() {
  const questions = listQuestions();
  const publicIds = publicResultIdSet();
  const groups = groupQuestions(questions);
  const anchorCount = questions.length; // derived from the doc, never hardcoded

  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Open questions
        </h1>
        <p className="max-w-2xl text-sm text-stone-600">
          The research hub — parsed live from{" "}
          <Link
            href="/docs/open_questions"
            className="underline decoration-stone-300 underline-offset-2 hover:text-stone-900"
          >
            docs/open_questions.md
          </Link>
          . Each question carries a one-line belief, a confidence level, and
          the experiments that bear on it. Evidence links jump to the
          relevant clean-result when public; otherwise to the gated task
          page.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <QuestionsBrowser />
          <span className="text-xs text-stone-500">
            {anchorCount} entries across {groups.length} groups
          </span>
        </div>
      </header>

      <div className="space-y-12">
        {groups.map((group) => (
          <GroupBlock key={group.name} group={group} publicIds={publicIds} />
        ))}
      </div>
    </article>
  );
}

function GroupBlock({
  group,
  publicIds,
}: {
  group: Group;
  publicIds: Set<number>;
}) {
  return (
    <section data-q-group={group.name} className="space-y-5">
      <h2 className="text-xl font-semibold tracking-tight">{group.name}</h2>
      {group.subgroups.map((sg, i) => (
        <div key={i} className="space-y-3">
          {sg.subsection && (
            <h3 className="text-sm font-medium uppercase tracking-wide text-stone-500">
              {sg.subsection}
            </h3>
          )}
          <ul className="divide-y divide-stone-200 overflow-hidden rounded-lg border border-stone-200 bg-white">
            {sg.questions.map((q) => (
              <li
                key={q.id}
                id={`q-${q.id}`}
                data-q-status={q.status}
                data-q-kind={q.kind}
                className="scroll-mt-4 space-y-2 px-4 py-3 sm:px-5 sm:py-4"
              >
                <QuestionRow question={q} publicIds={publicIds} />
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}

function QuestionRow({
  question,
  publicIds,
}: {
  question: Question;
  publicIds: Set<number>;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-xs text-stone-400">
          {question.kind === "application" ? `App ${question.number}` : question.number}
        </span>
        <h4 className="flex-1 text-sm font-medium leading-snug text-stone-900 sm:text-base">
          {question.title}
        </h4>
        {question.kind === "application" ? (
          <AppStatusBadge status={question.appStatus ?? "unknown"} />
        ) : (
          <QuestionMeta question={question} />
        )}
      </div>
      {question.belief && (
        <p className="text-sm leading-relaxed text-stone-700">
          <span className="text-stone-500">Belief: </span>
          {question.belief}
        </p>
      )}
      {question.next && (
        <p className="text-xs italic leading-relaxed text-stone-500">
          Next: {question.next}
        </p>
      )}
      {question.kind !== "application" && question.evidence.length > 0 && (
        <EvidenceList evidence={question.evidence} publicIds={publicIds} />
      )}
    </div>
  );
}

function QuestionMeta({ question }: { question: Question }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {question.confidence && <ConfidenceBadge confidence={question.confidence} />}
      <StatusBadge status={question.status} />
    </div>
  );
}

function ConfidenceBadge({ confidence }: { confidence: QuestionConfidence }) {
  const style = CONFIDENCE_STYLE[confidence];
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-medium ${style.chip}`}
    >
      <span className={`mr-1 inline-block h-1.5 w-1.5 rounded-full ${style.dot}`} />
      {confidence.charAt(0) + confidence.slice(1).toLowerCase()}
    </span>
  );
}

function StatusBadge({ status }: { status: "open" | "settled" }) {
  const cls =
    status === "settled"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : "border-stone-200 bg-stone-50 text-stone-600";
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-medium ${cls}`}
    >
      {status === "settled" ? "settled" : "open"}
    </span>
  );
}

function AppStatusBadge({ status }: { status: string }) {
  // Apps carry FREE-TEXT status (idea / tried / falsification risk / ...).
  // We don't try to enumerate the vocabulary; just chip the verbatim string.
  return (
    <span className="inline-flex items-center rounded border border-stone-200 bg-stone-50 px-2 py-0.5 text-[11px] font-medium text-stone-600">
      {status}
    </span>
  );
}

function EvidenceList({
  evidence,
  publicIds,
}: {
  evidence: number[];
  publicIds: Set<number>;
}) {
  return (
    <ul className="flex flex-wrap gap-x-2 gap-y-1 text-xs">
      <li className="text-stone-500">Evidence:</li>
      {evidence.map((id) => {
        const href = evidenceHrefForTaskId(id, publicIds);
        const isPublic = publicIds.has(id);
        return (
          <li key={id}>
            <Link
              href={href}
              // Disable prefetch on gated /tasks links so an idle hover on
              // the public /questions page doesn't trigger a server-side
              // render of the gated route.
              prefetch={isPublic ? undefined : false}
              className={
                "rounded font-mono text-stone-700 underline decoration-stone-300 underline-offset-2 hover:text-stone-900"
              }
            >
              #{id}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
