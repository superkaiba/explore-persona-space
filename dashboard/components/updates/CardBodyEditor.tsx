"use client";

import dynamic from "next/dynamic";
import { useCallback, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

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

  const onSave = useCallback(async () => {
    if (!editorRef.current || saving) return;
    const md = editorRef.current.getMarkdown();
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
  }, [saving, taskId, onSaved]);

  return (
    <div className="flex flex-col gap-3">
      {parseError && (
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

      <MDXEditorClient
        editorRef={editorRef}
        initialMarkdown={initialMarkdown}
        onError={(payload) => setParseError(payload.error)}
      />

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
