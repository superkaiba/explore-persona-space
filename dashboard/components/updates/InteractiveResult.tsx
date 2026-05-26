"use client";

import {
  isValidElement,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
  type ReactNode,
} from "react";
import Link from "next/link";
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  Maximize2,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  dayKey,
  formatTime,
  type CleanResult,
  type CleanResultDateField,
} from "@/lib/update-results";
import {
  ClaudeAskButton,
  ClaudeAskComposer,
  clearClaudeHover,
  dispatchClaudeHover,
  type ClaudeAskPayload,
} from "@/components/updates/MentorClaudePanel";
// The previous lift dropped the selection-popover comment flow that the
// /tasks/[id] page uses. We bring it back here in a card-shaped form via
// `CardCommentBox`, which wraps the body in an `AnchoredCommentsProvider`
// (from /app/tasks/[id]/AnchoredCommentsContext.tsx) and renders the same
// `CommentableBody` + `CommentList` widgets — so the Sagan-style drag-
// select → "+ Comment on selection" UX is reused unchanged. Comments
// persist into `tasks/<N>/comments.jsonl` via /api/updates/comment.
import { CardCommentBox } from "@/components/updates/CardCommentBox";
import { cn } from "@/lib/utils";

export function InteractiveResultCard({
  result,
  internal,
  dateField = "updatedAt",
  currentUserEmail = null,
}: {
  result: CleanResult;
  internal: boolean;
  dateField?: CleanResultDateField;
  currentUserEmail?: string | null;
}) {
  const [open, setOpen] = useState(false);
  const askPayload = useMemo(() => resultAskPayload(result), [result]);
  // Cards are 1:1 with `tasks/<N>/` and `tasks/<N>/` is keyed by GitHub
  // issue number, which lives on the result row as `githubIssueNumber`.
  // Cards without one (very old promotions) can't host comments.
  const taskId = result.githubIssueNumber ?? null;

  return (
    <>
      <article
        data-claude-anchor
        data-claude-scope-id={result.id}
        onMouseEnter={(event) => dispatchClaudeHover(askPayload, event.currentTarget)}
        onMouseLeave={(event) => clearClaudeHover(event.currentTarget)}
        onFocus={(event) => dispatchClaudeHover(askPayload, event.currentTarget)}
        onBlur={(event) => clearClaudeHover(event.currentTarget)}
        className="rounded-lg border border-border bg-panel p-8 transition-colors hover:bg-subtle/35 data-[claude-hovered=true]:border-accent data-[claude-hovered=true]:bg-accent/10"
      >
        <div className="flex items-start gap-6">
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="min-w-0 flex-1 text-left"
          >
            <div className="flex flex-wrap items-center gap-4 text-[11px] text-muted">
              <ResultBadge result={result} />
              {result.confidence && <span className="font-mono">{result.confidence}</span>}
              <span className="font-mono">{formatTime(result[dateField])}</span>
            </div>
            <h2 className="mt-4 text-[15px] font-semibold leading-snug text-fg">
              {result.title}
            </h2>
            {result.excerpt && (
              <p className="mt-5 text-[13px] leading-relaxed text-fg-soft">
                {result.excerpt}
              </p>
            )}
            <span className="mt-5 inline-flex items-center gap-1 text-[11px] text-muted">
              <Maximize2 className="h-3.5 w-3.5" />
              Open full result
            </span>
          </button>
          <ClaudeAskButton
            compact
            payload={askPayload}
            label={`Ask Claude about ${result.title}`}
            className="mt-0.5"
          />
        </div>
        {internal && <InternalLinks result={result} />}
        {taskId != null && (
          <div className="mt-6 border-t border-border pt-6">
            <CardCommentBox
              taskId={taskId}
              body={result.excerpt || result.title}
              currentUserEmail={currentUserEmail}
              layout="inline"
            />
          </div>
        )}
      </article>

      {open && (
        <ResultDetailOverlay
          result={result}
          internal={internal}
          dateField={dateField}
          currentUserEmail={currentUserEmail}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

export function InteractiveResultRow({
  result,
  internal,
  dateField = "updatedAt",
  currentUserEmail = null,
}: {
  result: CleanResult;
  internal: boolean;
  dateField?: CleanResultDateField;
  currentUserEmail?: string | null;
}) {
  const [open, setOpen] = useState(false);
  const askPayload = useMemo(() => resultAskPayload(result), [result]);

  return (
    <>
      <div
        data-claude-anchor
        data-claude-scope-id={result.id}
        onMouseEnter={(event) => dispatchClaudeHover(askPayload, event.currentTarget)}
        onMouseLeave={(event) => clearClaudeHover(event.currentTarget)}
        onFocus={(event) => dispatchClaudeHover(askPayload, event.currentTarget)}
        onBlur={(event) => clearClaudeHover(event.currentTarget)}
        className="rounded-md border border-border bg-panel px-3 py-2 transition-colors hover:bg-subtle/35 data-[claude-hovered=true]:border-accent data-[claude-hovered=true]:bg-accent/10"
      >
        <div className="flex min-w-0 items-start gap-3">
          <ResultBadge result={result} compact />
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="min-w-0 flex-1 text-left"
          >
            <div className="line-clamp-2 text-[13px] font-medium leading-snug text-fg">
              {result.title}
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-muted">
              {result.confidence && <span className="font-mono">{result.confidence}</span>}
              <span className="font-mono">{formatTime(result[dateField])}</span>
              <span className="inline-flex items-center gap-1">
                <Maximize2 className="h-3 w-3" />
                Open
              </span>
            </div>
          </button>
          <ClaudeAskButton
            compact
            payload={askPayload}
            label={`Ask Claude about ${result.title}`}
          />
        </div>
        {internal && <InternalLinks result={result} compact />}
      </div>

      {open && (
        <ResultDetailOverlay
          result={result}
          internal={internal}
          dateField={dateField}
          currentUserEmail={currentUserEmail}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

function ResultDetailOverlay({
  result,
  internal,
  dateField,
  currentUserEmail,
  onClose,
}: {
  result: CleanResult;
  internal: boolean;
  dateField: CleanResultDateField;
  currentUserEmail: string | null;
  onClose: () => void;
}) {
  const markdown = result.body || result.excerpt || "No result body is available.";
  const askPayload = useMemo(() => resultAskPayload(result), [result]);
  const contentRef = useRef<HTMLDivElement>(null);
  const taskId = result.githubIssueNumber ?? null;
  // Heading TOC fallback is only used on the no-taskId branch (where we
  // fall back to plain ReactMarkdown). When we have a task id we render
  // `CardCommentBox` instead, which uses `CommentableBody` without
  // heading IDs — the (small) cost of dropping the TOC is justified by
  // the much-larger gain of getting Sagan-style anchored comments.
  const headings = useMemo(() => extractMarkdownHeadings(markdown), [markdown]);
  const headingRenderCounts = new Map<string, number>();

  function scrollContentToId(id: string) {
    const container = contentRef.current;
    if (!container) return;
    if (!id) {
      container.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    const decodedId = decodeFragment(id);
    const candidateIds =
      decodedId.startsWith("user-content-")
        ? [decodedId, decodedId.replace(/^user-content-/, "")]
        : [decodedId];
    const target = Array.from(container.querySelectorAll<HTMLElement>("[id]")).find(
      (element) => candidateIds.includes(element.id),
    );
    if (!target) return;
    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    container.scrollTo({
      top: container.scrollTop + targetRect.top - containerRect.top - 14,
      behavior: "smooth",
    });
  }

  function handleAnchorClick(event: MouseEvent<HTMLAnchorElement>, href?: string) {
    if (!href?.startsWith("#")) return;
    event.preventDefault();
    scrollContentToId(href.slice(1));
  }

  // Suppress unused-var warning while keeping the symbol for the
  // no-taskId branch's TOC if we re-add one later.
  void headings;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/45 p-2 md:p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby={`result-${result.id}-title`}
      onClick={onClose}
    >
      <div
        className="flex flex-col overflow-hidden rounded-lg border border-border bg-panel shadow-rail"
        style={{
          width: "min(1400px, calc(100vw - 1rem))",
          height: "min(900px, calc(100dvh - 1rem))",
          minWidth: "min(720px, calc(100vw - 1rem))",
          minHeight: "min(520px, calc(100dvh - 1rem))",
          maxWidth: "calc(100vw - 1rem)",
          maxHeight: "calc(100dvh - 1rem)",
          resize: "both",
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-start gap-3 border-b border-border px-8 py-6">
          <div className="min-w-0 flex-1">
            <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px] text-muted">
              <ResultBadge result={result} />
              {result.confidence && <span className="font-mono">{result.confidence}</span>}
              <span className="font-mono">
                {dayKey(result[dateField])} {formatTime(result[dateField])}
              </span>
            </div>
            <h2
              id={`result-${result.id}-title`}
              className="text-[16px] font-semibold leading-snug text-fg"
            >
              {result.title}
            </h2>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <ClaudeAskButton payload={askPayload} label="Ask Claude Code" />
            <button
              type="button"
              onClick={onClose}
              className="rounded-md p-1.5 text-muted hover:bg-subtle hover:text-fg"
              aria-label="Close result"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-hidden">
          <div className="grid h-full min-h-0 space-y-8 overflow-y-auto px-8 py-6 lg:overflow-y-auto">
            <ClaudeAskComposer
              payload={askPayload}
              placeholder="Ask Claude Code to inspect this result..."
              className="bg-subtle/30"
            />
            <div
              ref={contentRef}
              className="prose prose-sm max-w-none prose-headings:text-fg prose-p:text-fg-soft prose-strong:text-fg prose-code:text-fg prose-pre:border prose-pre:border-border prose-pre:bg-subtle prose-li:text-fg-soft prose-a:text-accent"
            >
              {taskId != null ? (
                <CardCommentBox
                  taskId={taskId}
                  body={markdown}
                  currentUserEmail={currentUserEmail}
                  layout="rail"
                />
              ) : (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        onClick={(event) => handleAnchorClick(event, href)}
                        target={href?.startsWith("#") ? undefined : "_blank"}
                        rel={href?.startsWith("#") ? undefined : "noopener noreferrer"}
                        className="text-accent hover:underline"
                      >
                        {children}
                      </a>
                    ),
                    h1: ({ children }) => renderMarkdownHeading(1, children, headingRenderCounts),
                    h2: ({ children }) => renderMarkdownHeading(2, children, headingRenderCounts),
                    h3: ({ children }) => renderMarkdownHeading(3, children, headingRenderCounts),
                    h4: ({ children }) => renderMarkdownHeading(4, children, headingRenderCounts),
                    h5: ({ children }) => renderMarkdownHeading(5, children, headingRenderCounts),
                    h6: ({ children }) => renderMarkdownHeading(6, children, headingRenderCounts),
                  }}
                >
                  {markdown}
                </ReactMarkdown>
              )}
            </div>
          </div>
        </div>

        {internal && (
          <footer className="flex flex-wrap items-center gap-3 border-t border-border px-8 py-6 text-[12px] text-muted">
            <Link href={result.href} className="inline-flex items-center gap-1 hover:text-fg">
              Open claim
              <ExternalLink className="h-3 w-3" />
            </Link>
            {result.githubIssueNumber != null && (
              <a
                href={`https://github.com/superkaiba/explore-persona-space/issues/${result.githubIssueNumber}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 hover:text-fg"
              >
                GitHub #{result.githubIssueNumber}
                <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </footer>
        )}
      </div>
    </div>
  );
}

function ResultBadge({ result, compact = false }: { result: CleanResult; compact?: boolean }) {
  const Icon = result.useful ? CheckCircle2 : AlertTriangle;
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md border px-1.5 py-0.5 font-medium",
        compact ? "text-[10px]" : "text-[11px]",
        result.useful
          ? "border-confidence-high/35 bg-confidence-high/10 text-confidence-high"
          : "border-confidence-low/40 bg-confidence-low/10 text-muted",
      )}
    >
      <Icon className={compact ? "h-3 w-3" : "h-3.5 w-3.5"} />
      {result.useful ? "Useful" : "Not useful"}
    </span>
  );
}

function InternalLinks({
  result,
  compact = false,
}: {
  result: CleanResult;
  compact?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-3 text-[11px] text-muted",
        compact ? "mt-2 pl-16" : "mt-3",
      )}
    >
      <Link href={result.href} className="hover:text-fg">
        Open claim
      </Link>
      {result.githubIssueNumber != null && (
        <a
          href={`https://github.com/superkaiba/explore-persona-space/issues/${result.githubIssueNumber}`}
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-fg"
        >
          GitHub #{result.githubIssueNumber}
        </a>
      )}
    </div>
  );
}

function resultAskPayload(result: CleanResult): ClaudeAskPayload {
  return {
    scopeKind: "result",
    scopeId: result.id,
    sourceLabel:
      result.githubIssueNumber == null
        ? `Claim ${result.id.slice(0, 8)}`
        : `Issue #${result.githubIssueNumber}`,
    scopeTitle: result.title,
    contextMd: resultContext(result),
    suggestedQuestion: "Inspect this result and explain what it means.",
  };
}

type MarkdownHeading = {
  id: string;
  text: string;
  depth: number;
  index: number;
};

function extractMarkdownHeadings(markdown: string): MarkdownHeading[] {
  const counts = new Map<string, number>();
  const headings: MarkdownHeading[] = [];
  const pattern = /^(#{1,6})\s+(.+?)\s*#*\s*$/gm;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(markdown))) {
    const text = plainMarkdownText(match[2]);
    if (!text) continue;
    const baseId = githubLikeSlug(text);
    const id = dedupeSlug(baseId, counts);
    headings.push({
      id,
      text,
      depth: match[1].length,
      index: headings.length,
    });
  }

  return headings;
}

function renderMarkdownHeading(
  depth: 1 | 2 | 3 | 4 | 5 | 6,
  children: ReactNode,
  counts: Map<string, number>,
) {
  const text = nodeText(children);
  const id = dedupeSlug(githubLikeSlug(text), counts);
  const className = "scroll-mt-4";

  if (depth === 1) return <h1 id={id} className={className}>{children}</h1>;
  if (depth === 2) return <h2 id={id} className={className}>{children}</h2>;
  if (depth === 3) return <h3 id={id} className={className}>{children}</h3>;
  if (depth === 4) return <h4 id={id} className={className}>{children}</h4>;
  if (depth === 5) return <h5 id={id} className={className}>{children}</h5>;
  return <h6 id={id} className={className}>{children}</h6>;
}

function nodeText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return nodeText(node.props.children);
  return "";
}

function plainMarkdownText(value: string) {
  return value
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[([^\]]*)]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/[*_~]/g, "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function githubLikeSlug(value: string) {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, "")
    .replace(/\s+/g, "-");
  return slug || "section";
}

function dedupeSlug(baseId: string, counts: Map<string, number>) {
  const count = counts.get(baseId) ?? 0;
  counts.set(baseId, count + 1);
  return count === 0 ? baseId : `${baseId}-${count}`;
}

function decodeFragment(value: string) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function resultContext(result: CleanResult) {
  const resultUrl = result.href.startsWith("http")
    ? result.href
    : `https://dashboard.superkaiba.com${result.href}`;
  const inspectionHint = result.href.startsWith("http")
    ? "Inspect the full GitHub issue body and linked artifacts before answering. Use curl/git/shell tools to inspect or download referenced artifacts."
    : "Inspect the full result yourself before answering. Use psql against $DASHBOARD_DATABASE_URL to fetch claim.body_json and comments for this claim id, and use curl/git/shell tools to inspect or download referenced artifacts.";
  return [
    "Claude Code inspection target:",
    `Title: ${result.title}`,
    `Result ID: ${result.id}`,
    `Classification: ${result.useful ? "useful" : "not useful"}`,
    `Confidence: ${result.confidence ?? "not set"}`,
    `GitHub issue created: ${dayKey(result.createdAt)} ${formatTime(result.createdAt)}`,
    `Dashboard updated: ${dayKey(result.updatedAt)} ${formatTime(result.updatedAt)}`,
    `Result URL: ${resultUrl}`,
    result.githubIssueNumber == null
      ? null
      : `GitHub issue: https://github.com/superkaiba/explore-persona-space/issues/${result.githubIssueNumber}`,
    "",
    inspectionHint,
    "",
    "Full result:",
    "Not included in this prompt; fetch it from the dashboard database or linked artifacts when needed.",
    "",
    "Excerpt:",
    result.excerpt || "No excerpt available.",
  ]
    .filter(Boolean)
    .join("\n");
}

