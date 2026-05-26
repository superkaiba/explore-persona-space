"use client";

import Link from "next/link";
import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { addComment, askClaude } from "./comment-actions";
import { useAnchoredComments } from "./AnchoredCommentsContext";

type CommentKind = "question" | "note";

export function CommentForm({
  taskId,
  editorEnabled,
  editorAuthed,
}: {
  taskId: number;
  editorEnabled: boolean;
  editorAuthed: boolean;
}) {
  const [kind, setKind] = useState<CommentKind>("note");
  const [body, setBody] = useState("");
  const [status, setStatus] = useState<
    { kind: "ok"; text: string } | { kind: "err"; text: string } | null
  >(null);
  const [pending, startTransition] = useTransition();
  const router = useRouter();
  const { pendingQuote, setPendingQuote } = useAnchoredComments();

  if (!editorEnabled) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-3 text-sm text-stone-500">
        Comments are disabled. Set <code>SITE_PASSWORD</code> in the
        dashboard&apos;s environment to enable.
      </p>
    );
  }

  if (!editorAuthed) {
    return (
      <p className="rounded border border-dashed border-stone-300 bg-white px-4 py-3 text-sm text-stone-700">
        <Link
          href={`/sign-in?next=${encodeURIComponent(`/tasks/${taskId}`)}`}
          className="font-medium underline"
        >
          Sign in
        </Link>{" "}
        to post a comment or ask Claude Code a question.
      </p>
    );
  }

  function buildFormData(): FormData {
    const fd = new FormData();
    fd.set("taskId", String(taskId));
    fd.set("kind", kind);
    fd.set("body", body);
    if (pendingQuote) fd.set("anchorQuote", pendingQuote);
    return fd;
  }

  function onPost() {
    if (!body.trim() || pending) return;
    setStatus(null);
    startTransition(async () => {
      const result = await addComment(buildFormData());
      if (result.ok) {
        setStatus({ kind: "ok", text: `Posted as ${result.commentId}.` });
        setBody("");
        setPendingQuote(null);
        router.refresh();
      } else {
        setStatus({ kind: "err", text: result.error });
      }
    });
  }

  function onAskClaude() {
    if (!body.trim() || pending) return;
    setStatus(null);
    const fd = new FormData();
    fd.set("taskId", String(taskId));
    fd.set("body", body);
    if (pendingQuote) fd.set("anchorQuote", pendingQuote);
    startTransition(async () => {
      const result = await askClaude(fd);
      if (result.ok) {
        const sessionNote = result.spawnedSessionId
          ? ` Spawned Claude Code session ${result.spawnedSessionId}.`
          : "";
        setStatus({
          kind: "ok",
          text:
            `Posted question as ${result.commentId}.${sessionNote} ` +
            `Answer will appear here once Claude responds.`,
        });
        setBody("");
        setKind("note");
        setPendingQuote(null);
        router.refresh();
      } else {
        setStatus({ kind: "err", text: result.error });
      }
    });
  }

  return (
    <div className="space-y-2 rounded border border-stone-200 bg-white p-3">
      <div className="flex items-center gap-2 text-xs text-stone-500">
        <label className="flex items-center gap-1">
          <span>kind:</span>
          <select
            value={kind}
            onChange={(e) => setKind(e.target.value as CommentKind)}
            disabled={pending}
            className="rounded border border-stone-300 bg-white px-1.5 py-0.5"
          >
            <option value="note">note</option>
            <option value="question">question</option>
          </select>
        </label>
        <span className="text-stone-400">
          (markdown — code fences + links supported)
        </span>
      </div>

      {pendingQuote && (
        <div className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-2 py-1.5 text-xs">
          <div className="flex-1">
            <div className="font-medium text-amber-900">
              Commenting on selection:
            </div>
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
      )}

      <textarea
        value={body}
        onChange={(e) => {
          setBody(e.target.value);
          setStatus(null);
        }}
        disabled={pending}
        placeholder={
          pendingQuote
            ? "Write a comment on this selection (markdown)."
            : kind === "question"
              ? "Ask a question. Hit “Ask Claude Code” to spawn a session that searches the codebase + eval results and posts an answer."
              : "Add a note (markdown). Select text in the body above to anchor the comment to that span."
        }
        rows={5}
        className="w-full resize-y rounded border border-stone-300 bg-white px-2 py-1.5 text-sm font-mono"
      />

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onPost}
          disabled={pending || !body.trim()}
          className="rounded bg-stone-900 px-3 py-1.5 text-sm font-medium text-white disabled:bg-stone-300"
        >
          {pending ? "…" : `Post ${kind}`}
        </button>
        {kind === "question" && (
          <button
            type="button"
            onClick={onAskClaude}
            disabled={pending || !body.trim()}
            className="rounded border border-stone-900 bg-white px-3 py-1.5 text-sm font-medium text-stone-900 disabled:opacity-50"
            title="Posts the question AND spawns a Claude Code session that searches the codebase + eval results to answer."
          >
            {pending ? "…" : "Ask Claude Code"}
          </button>
        )}
        {status && (
          <span
            className={
              status.kind === "ok"
                ? "text-sm text-emerald-700"
                : "text-sm text-red-700"
            }
          >
            {status.text}
          </span>
        )}
      </div>
    </div>
  );
}
