"use client";

/**
 * DocBody — the client shell for a /docs/<slug> page.
 *
 * Wires the doc body + its anchored comments into the shared <MarkdownDoc>
 * keystone instead of the retired standalone DocComments component:
 *
 *   - <AnchoredCommentsProvider> supplies the committed comment anchors
 *     (rows that carry a `quote`) so MarkdownDoc wraps each in <mark> on the
 *     rendered body — the SAME highlight-to-comment surface tasks use.
 *   - <MarkdownDoc showToc enableAskClaude docId={slug}> renders the body
 *     (auto-TOC + per-header collapse + Ask-Claude). NOT public: docs are
 *     read-gated, so comment writes + Ask-Claude are enabled.
 *   - <DocCommentRail> reads the pending selection from context and POSTs to
 *     the existing /api/docs/comment API (comments stay gitignored — the
 *     route writes to the doc's `.comments/<stem>.jsonl`). It also lists
 *     open/addressed comments, lets the author delete, and triggers the
 *     "Address all" Claude rewrite via /api/docs/address-comments.
 *
 * Selection -> popover -> pendingQuote -> rail composer is the highlight
 * flow; the rail mirrors the task CommentForm's pendingQuote handling.
 */
import { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  AnchoredCommentsProvider,
  useAnchoredComments,
  type AnchorRecord,
} from "@/app/tasks/[id]/AnchoredCommentsContext";
import { MarkdownDoc } from "@/components/MarkdownDoc";

export type DocCommentView = {
  id: string;
  ts: string;
  author: string;
  kind: "doc-comment" | "doc-comment-reply";
  body: string;
  section_label?: string;
  quote?: string;
  in_reply_to?: string;
  addressed?: boolean;
  addressed_note?: string;
};

export function DocBody({
  slug,
  body,
  title,
  initialComments,
  editorAuthed,
  currentUserEmail,
}: {
  slug: string;
  body: string;
  title: string;
  initialComments: DocCommentView[];
  editorAuthed: boolean;
  currentUserEmail: string | null;
}) {
  const [comments, setComments] = useState<DocCommentView[]>(initialComments);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`/api/docs/comment?slug=${encodeURIComponent(slug)}`, {
        cache: "no-store",
      });
      const data = await res.json();
      if (data.ok) setComments(data.comments as DocCommentView[]);
    } catch {
      /* leave existing list; transient fetch error */
    }
  }, [slug]);

  // Committed anchors: open root comments that carry a `quote`. MarkdownDoc
  // wraps each `quote` occurrence in <mark data-comment-id> on the body.
  const anchors: AnchorRecord[] = useMemo(
    () =>
      comments
        .filter(
          (c) =>
            c.kind === "doc-comment" &&
            !c.addressed &&
            typeof c.quote === "string" &&
            c.quote.trim().length >= 4,
        )
        .map((c) => ({ id: c.id, quote: (c.quote as string).trim() })),
    [comments],
  );

  // Inline-composer create hook: POST the anchored comment to the docs API +
  // refetch. Wired into MarkdownDoc so highlight-to-comment opens the inline
  // composer at the selection (works at any width). The rail keeps the comment
  // list + whole-doc composer + "Address all".
  const onCommentCreate = useCallback(
    async ({ quote, body: text }: { quote: string; body: string }): Promise<boolean> => {
      if (!editorAuthed) return false;
      const trimmed = text.trim();
      if (!trimmed) return false;
      try {
        const res = await fetch("/api/docs/comment", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ slug, body: trimmed, quote: quote.trim() || undefined }),
        });
        const data = await res.json();
        if (!data.ok) return false;
        await refresh();
        return true;
      } catch {
        return false;
      }
    },
    [slug, editorAuthed, refresh],
  );

  return (
    <AnchoredCommentsProvider
      anchors={anchors}
      onCommentCreate={editorAuthed ? onCommentCreate : null}
    >
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <MarkdownDoc
            body={body}
            showToc
            enableCollapsibleSections
            enableAskClaude
            askClaudeTitle={title}
            docId={slug}
            onCommentCreate={editorAuthed ? onCommentCreate : undefined}
          />
        </div>
        <aside className="lg:sticky lg:top-4 lg:self-start">
          <DocCommentRail
            slug={slug}
            comments={comments}
            editorAuthed={editorAuthed}
            currentUserEmail={currentUserEmail}
            onChanged={refresh}
          />
        </aside>
      </div>
    </AnchoredCommentsProvider>
  );
}

function DocCommentRail({
  slug,
  comments,
  editorAuthed,
  currentUserEmail,
  onChanged,
}: {
  slug: string;
  comments: DocCommentView[];
  editorAuthed: boolean;
  currentUserEmail: string | null;
  onChanged: () => void | Promise<void>;
}) {
  const router = useRouter();
  const { pendingQuote, setPendingQuote, setHoveredId, requestScrollTo } =
    useAnchoredComments();
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [addressing, setAddressing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAddressed, setShowAddressed] = useState(false);

  const open = comments.filter((c) => c.kind === "doc-comment" && !c.addressed);
  const addressed = comments.filter((c) => c.kind === "doc-comment" && c.addressed);

  async function submit() {
    if (!draft.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/docs/comment", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          slug,
          body: draft,
          quote: pendingQuote?.trim() || undefined,
        }),
      });
      const data = await res.json();
      if (!data.ok) {
        setError(data.error === "unauthorized" ? "Sign in to comment." : data.error || "failed");
      } else {
        setDraft("");
        setPendingQuote(null);
        await onChanged();
      }
    } catch {
      setError("network error");
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    try {
      await fetch("/api/docs/comment", {
        method: "DELETE",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ slug, commentId: id }),
      });
      await onChanged();
    } catch {
      /* ignore; next refresh reconciles */
    }
  }

  async function addressAll() {
    if (open.length === 0 || addressing) return;
    setAddressing(true);
    setError(null);
    try {
      const res = await fetch("/api/docs/address-comments", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ slug }),
      });
      const data = await res.json();
      if (!data.ok) {
        setError(
          data.error === "editor cookie required"
            ? "Editor access required to auto-address."
            : data.error || "failed",
        );
        setAddressing(false);
      } else {
        // The doc on disk changed; reload to render the revised version.
        router.refresh();
      }
    } catch {
      setError("network error");
      setAddressing(false);
    }
  }

  return (
    <section className="rounded-lg border border-stone-200 bg-white p-4 text-sm">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-stone-500">
          Comments {open.length > 0 && <span className="text-stone-400">({open.length} open)</span>}
        </h2>
        <button
          type="button"
          onClick={addressAll}
          disabled={open.length === 0 || addressing}
          className="rounded bg-stone-900 px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-300"
        >
          {addressing ? "Addressing…" : `Address all${open.length ? ` (${open.length})` : ""}`}
        </button>
      </div>

      {editorAuthed ? (
        <div className="mt-3 space-y-2">
          {pendingQuote ? (
            <div className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-2 py-1.5 text-xs">
              <div className="flex-1">
                <div className="font-medium text-amber-900">Commenting on selection:</div>
                <blockquote className="mt-0.5 line-clamp-3 italic text-amber-950">
                  “{pendingQuote.length > 200 ? pendingQuote.slice(0, 200) + "…" : pendingQuote}”
                </blockquote>
              </div>
              <button
                type="button"
                onClick={() => setPendingQuote(null)}
                className="text-amber-700 hover:text-amber-900"
                aria-label="Clear anchor"
                title="Clear anchor"
              >
                ✕
              </button>
            </div>
          ) : (
            <p className="text-[11px] text-stone-400">
              Select text in the doc to anchor a comment, or comment on the whole document.
            </p>
          )}
          <textarea
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              setError(null);
            }}
            placeholder={
              pendingQuote ? "Comment on this selection…" : "Leave a comment on this document…"
            }
            rows={3}
            className="w-full rounded border border-stone-300 px-2 py-1.5 text-sm text-stone-800 placeholder:text-stone-400"
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-rose-600">{error}</span>
            <button
              type="button"
              onClick={submit}
              disabled={!draft.trim() || busy}
              className="rounded border border-stone-300 px-3 py-1 text-xs font-medium text-stone-700 transition-colors hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? "Saving…" : "Comment"}
            </button>
          </div>
        </div>
      ) : (
        <p className="mt-3 rounded border border-dashed border-stone-300 px-3 py-2 text-xs text-stone-500">
          <Link
            href={`/sign-in?next=${encodeURIComponent(`/docs/${slug}`)}`}
            className="font-medium underline"
          >
            Sign in
          </Link>{" "}
          to comment.
        </p>
      )}

      {open.length > 0 && (
        <ul className="mt-4 space-y-3 border-t border-stone-100 pt-3">
          {open.map((c) => (
            <li
              key={c.id}
              className="space-y-1"
              onMouseEnter={() => c.quote && setHoveredId(c.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              {c.section_label && (
                <span className="inline-block rounded bg-stone-100 px-1.5 py-0.5 text-[11px] font-medium text-stone-600">
                  {c.section_label}
                </span>
              )}
              {c.quote && (
                <button
                  type="button"
                  onClick={() => requestScrollTo(c.id)}
                  className="block w-full border-l-2 border-amber-300 pl-2 text-left text-xs italic text-stone-500 hover:text-stone-700"
                  title="Scroll to highlighted text"
                >
                  “{c.quote.length > 160 ? c.quote.slice(0, 160) + "…" : c.quote}”
                </button>
              )}
              <p className="whitespace-pre-wrap text-stone-800">{c.body}</p>
              <div className="flex items-center gap-2 text-[11px] text-stone-400">
                <span>{c.author}</span>
                {currentUserEmail && c.author === currentUserEmail && (
                  <button
                    type="button"
                    onClick={() => remove(c.id)}
                    className="hover:text-rose-600"
                  >
                    delete
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      {addressed.length > 0 && (
        <div className="mt-4 border-t border-stone-100 pt-3">
          <button
            type="button"
            onClick={() => setShowAddressed((v) => !v)}
            className="text-xs font-medium text-stone-500 hover:text-stone-800"
          >
            {showAddressed ? "▾" : "▸"} Addressed ({addressed.length})
          </button>
          {showAddressed && (
            <ul className="mt-2 space-y-2">
              {addressed.map((c) => (
                <li key={c.id} className="space-y-0.5 text-xs text-stone-500">
                  {c.section_label && <span className="font-medium">{c.section_label}: </span>}
                  <span className="line-through">{c.body}</span>
                  {c.addressed_note && (
                    <p className="not-italic text-emerald-700">→ {c.addressed_note}</p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
