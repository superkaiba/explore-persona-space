"use client";

import dynamic from "next/dynamic";
import { useCallback, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

import { MDXEditorErrorBoundary } from "./MDXEditorErrorBoundary";

/**
 * Inline WYSIWYG editor for clean-result bodies (`/updates` modal view).
 *
 * Mounts `@mdxeditor/editor`, which parses markdown -> MDAST -> Lexical and
 * serializes Lexical -> MDAST -> markdown via `mdast-util-to-markdown`. The
 * round-trip is lossless for GFM tables, fenced code, links, lists,
 * blockquotes, headings, inline formatting, and thematic breaks — i.e.
 * everything the old TipTap StarterKit + tiptap-markdown stack rejected.
 *
 * Bundle hygiene: this thin wrapper is a Client Component that
 * dynamically imports `./CardBodyEditorClient` with `ssr:false`, which
 * in turn imports `@mdxeditor/editor` + its stylesheet. MDXEditor pulls
 * in Lexical + CodeMirror; keeping it behind `next/dynamic` keeps the
 * /updates initial bundle lean and prevents the SSR pass from touching
 * `window`.
 *
 * Save flow: on click, we read markdown from the editor ref via
 * `MDXEditorMethods.getMarkdown()` and POST it to `/api/updates/body`,
 * unchanged from the TipTap version (no API shape change).
 *
 * Auth: parent (InteractiveResultCard) only renders this when the
 * server-rendered `canEdit` flag (isEditorAuthed()) is true. The save
 * route re-checks the editor cookie before writing.
 */

// Type-only import from the lib — does NOT pull the runtime into this
// bundle (TS strips type imports). The actual MDXEditor + plugin code
// lives behind the next/dynamic boundary in `./CardBodyEditorClient`.
import type { MDXEditorMethods } from "@mdxeditor/editor";

const MDXEditorClient = dynamic(() => import("./CardBodyEditorClient"), {
  ssr: false,
  loading: () => (
    <div className="rounded border border-border bg-canvas px-4 py-6 text-sm text-muted">
      Loading editor…
    </div>
  ),
});

export function CardBodyEditor({
  taskId,
  initialMarkdown,
  onSaved,
  onCancel,
}: {
  taskId: number;
  initialMarkdown: string;
  onSaved: (newMarkdown: string) => void;
  onCancel: () => void;
}) {
  const editorRef = useRef<MDXEditorMethods>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  // When the rich editor can't load this body (recognized parse error
  // via onError, OR a render-phase crash caught by the error boundary),
  // we drop to a raw-markdown textarea. `active` swaps the editor for
  // the textarea; `reason` is the message shown in the notice.
  const [rawFallback, setRawFallback] = useState<{
    active: boolean;
    reason: string;
  } | null>(null);
  // Controlled value for the raw-markdown textarea, seeded from the
  // original source. This is the safe, deterministic source we POST in
  // fallback mode — we do NOT rely on editorRef.getMarkdown() (which,
  // on a recognized parse error, returns the original source anyway:
  // the catch sets markdown$ = markdownValue). Reading rawValue removes
  // any ambiguity about partial-vs-original content.
  const [rawValue, setRawValue] = useState(initialMarkdown);

  const onSave = useCallback(async () => {
    if (saving) return;
    // Read the markdown from whichever editor is ACTIVE. In fallback
    // mode the rich editor isn't mounted, so editorRef is null and the
    // textarea's controlled value (rawValue) is the source of truth.
    let md: string;
    if (rawFallback?.active) {
      md = rawValue;
    } else {
      if (!editorRef.current) return;
      md = editorRef.current.getMarkdown();
    }
    setSaving(true);
    setSaveError(null);
    try {
      const res = await fetch("/api/updates/body", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ taskId, body: md }),
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
          const j = (await res.json()) as { error?: string };
          if (j.error) msg = j.error;
        } catch {
          // body wasn't JSON; keep the HTTP-code message.
        }
        setSaveError(msg);
        return;
      }
      onSaved(md);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [saving, taskId, onSaved, rawFallback, rawValue]);

  return (
    <div className="flex flex-col gap-3">
      {rawFallback?.active && (
        <div className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <div className="font-medium">
            Switched to raw-markdown editing.
          </div>
          <p className="mt-1">
            The rich editor couldn&apos;t parse this body, so it&apos;s
            showing the raw markdown source instead. Save still works —
            it writes whatever&apos;s in the box below. {rawFallback.reason}{" "}
            You can also edit at{" "}
            <a
              href={`/tasks/${taskId}/edit`}
              className="underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              /tasks/{taskId}/edit
            </a>
            .
          </p>
        </div>
      )}

      {/*
        Amber parse notice shown only when a RECOGNIZED parse error fired
        but we have NOT yet dropped to the textarea (defensive — onError
        now also flips rawFallback, so in practice the rawFallback notice
        above takes over. Kept as a belt-and-suspenders signal so a
        parseError without a fallback never goes unexplained).
      */}
      {parseError && !rawFallback?.active && (
        <div className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <div className="font-medium">
            MDXEditor could not parse part of this body cleanly.
          </div>
          <p className="mt-1">
            {parseError} If saving from here drops content, fall back to
            the raw markdown editor at{" "}
            <a
              href={`/tasks/${taskId}/edit`}
              className="underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              /tasks/{taskId}/edit
            </a>{" "}
            instead.
          </p>
        </div>
      )}

      {rawFallback?.active ? (
        // Raw-markdown fallback: render a controlled textarea INSTEAD OF
        // the MDXEditor subtree. We must NOT mount MDXEditorClient here —
        // mounting it again would re-trigger the same crash. Styling
        // mirrors the rich editor surface (bordered, full-width,
        // monospace, comparable min-height).
        <textarea
          value={rawValue}
          onChange={(e) => setRawValue(e.target.value)}
          spellCheck={false}
          className={
            "w-full min-h-[280px] resize-y rounded border border-border " +
            "bg-canvas px-4 py-3 font-mono text-sm text-fg " +
            "focus:outline-none focus:ring-1 focus:ring-accent"
          }
        />
      ) : (
        // Happy path: the rich editor, wrapped in an error boundary that
        // catches render-phase throws (path 2) and flips the SAME
        // fallback flag onError uses for recognized parse errors (path
        // 1). Once fallbackActive is true we stop rendering this branch
        // entirely, so MDXEditorClient is never mounted in fallback mode.
        <MDXEditorErrorBoundary
          fallbackActive={false}
          onCrash={(msg) => setRawFallback({ active: true, reason: msg })}
        >
          <MDXEditorClient
            editorRef={editorRef}
            initialMarkdown={initialMarkdown}
            onError={(payload) => {
              // Recognized parse error: keep the amber notice for
              // context AND drop to the textarea. The rich editor body
              // is empty/lossy in this state, so surfacing the raw
              // source is strictly better than leaving the user on a
              // blank rich editor.
              setParseError(payload.error);
              setRawFallback({ active: true, reason: payload.error });
            }}
          />
        </MDXEditorErrorBoundary>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void onSave()}
          disabled={saving}
          className="inline-flex items-center gap-1.5 rounded bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:bg-stone-300"
        >
          {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {saving ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={saving}
          className="rounded border border-border bg-panel px-3 py-1.5 text-sm font-medium text-fg hover:bg-subtle disabled:opacity-50"
        >
          Cancel
        </button>
        {saveError && (
          <span className="text-xs text-red-700">{saveError}</span>
        )}
      </div>
    </div>
  );
}
