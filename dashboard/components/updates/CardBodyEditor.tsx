"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { EditorContent, useEditor, type Editor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Link from "@tiptap/extension-link";
import Placeholder from "@tiptap/extension-placeholder";
import { Markdown, type MarkdownStorage } from "tiptap-markdown";
import {
  Bold,
  Code,
  Heading1,
  Heading2,
  Heading3,
  Italic,
  Link as LinkIcon,
  List,
  ListOrdered,
  Quote,
  Redo,
  Strikethrough,
  Undo,
} from "lucide-react";

/**
 * Inline WYSIWYG editor for clean-result bodies (`/updates` modal view).
 *
 * Mounts TipTap with the StarterKit + Link + Placeholder + tiptap-markdown
 * extensions. Seeded with the current markdown body. On Save: serializes
 * back to markdown via `editor.storage.markdown.getMarkdown()` and POSTs to
 * /api/updates/body, which shells out to `scripts/task.py set-body` on the
 * server (flock + git commit).
 *
 * Round-trip caveats (tiptap-markdown / StarterKit support):
 *   - Tables, footnotes, raw HTML, images-with-titles, and definition lists
 *     are NOT in StarterKit. If the source body uses any of those, we
 *     surface a warning toast when entering edit mode and disable Save.
 *   - GFM strikethrough + task lists ship in StarterKit + tiptap-markdown.
 *
 * Auth: the parent (InteractiveResultCard) only renders this when the
 * server-rendered `canEdit` flag (isEditorAuthed()) is true. The server
 * route re-checks the editor cookie before writing.
 */

const UNSUPPORTED_MARKDOWN_PATTERNS: Array<{ pattern: RegExp; label: string }> = [
  // GFM tables: `|` columns + a `---|---` separator row.
  { pattern: /^\s*\|.+\|\s*\n\s*\|[\s:|-]+\|\s*$/m, label: "tables" },
  // Footnote definitions / references.
  { pattern: /\[\^[^\]]+\]/, label: "footnotes" },
  // Definition lists (Pandoc-style).
  { pattern: /^[^\n]+\n:\s+/m, label: "definition lists" },
  // Raw HTML block tags we don't round-trip cleanly. Keep this list short
  // — most prose bodies have none, and false-positive on inline tags would
  // be annoying.
  { pattern: /<(table|tbody|thead|tr|td|th|details|summary|iframe)\b/i, label: "raw HTML blocks" },
];

function detectUnsupportedMarkdown(md: string): string[] {
  const hits = new Set<string>();
  for (const { pattern, label } of UNSUPPORTED_MARKDOWN_PATTERNS) {
    if (pattern.test(md)) hits.add(label);
  }
  return Array.from(hits);
}

/**
 * tiptap-markdown attaches a `MarkdownStorage` instance under
 * `editor.storage.markdown` at runtime, but `editor.storage` is typed as
 * a generic `Record<string, unknown>`-shaped index map so TS can't see
 * the field. Cast through `unknown` to a partial shape and guard the
 * getter so we degrade gracefully if the extension is ever swapped out.
 */
function getMarkdownFromEditor(editor: Editor): string {
  const storage = editor.storage as unknown as {
    markdown?: Partial<MarkdownStorage>;
  };
  const fn = storage.markdown?.getMarkdown;
  if (typeof fn !== "function") return "";
  try {
    return fn.call(storage.markdown);
  } catch {
    return "";
  }
}

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
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const unsupported = useMemo(
    () => detectUnsupportedMarkdown(initialMarkdown),
    [initialMarkdown],
  );
  const disabled = unsupported.length > 0;

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        // tiptap-markdown handles the markdown <-> doc mapping; disable
        // StarterKit's own codeBlock so we don't double-register.
        codeBlock: { HTMLAttributes: { class: "rounded bg-stone-900 p-3 text-stone-50" } },
      }),
      Link.configure({
        openOnClick: false,
        HTMLAttributes: { class: "text-accent underline" },
      }),
      Placeholder.configure({
        placeholder: "Write the result body in plain English…",
      }),
      Markdown.configure({
        html: false,
        tightLists: true,
        bulletListMarker: "-",
        linkify: false,
        breaks: false,
        transformPastedText: true,
        transformCopiedText: true,
      }),
    ],
    content: initialMarkdown,
    editable: !disabled,
    immediatelyRender: false,
    editorProps: {
      attributes: {
        class:
          "prose prose-sm max-w-none focus:outline-none min-h-[280px] px-4 py-3 " +
          "prose-headings:text-fg prose-p:text-fg-soft prose-strong:text-fg " +
          "prose-code:text-fg prose-pre:border prose-pre:border-border " +
          "prose-pre:bg-subtle prose-li:text-fg-soft prose-a:text-accent",
      },
    },
  });

  useEffect(() => {
    // Re-seed if the underlying markdown changes (rare: parent swaps it
    // after a save round-trip from elsewhere). We compare against the
    // editor's current markdown to avoid stomping on in-flight typing.
    if (!editor) return;
    const current = getMarkdownFromEditor(editor);
    if (current.trim() === initialMarkdown.trim()) return;
    editor.commands.setContent(initialMarkdown, { emitUpdate: false });
  }, [editor, initialMarkdown]);

  const onSave = useCallback(async () => {
    if (!editor || saving) return;
    const md = getMarkdownFromEditor(editor);
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
  }, [editor, saving, taskId, onSaved]);

  return (
    <div className="flex flex-col gap-3">
      {disabled && (
        <div className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <div className="font-medium">
            This body uses markdown features the editor does not round-trip
            cleanly: {unsupported.join(", ")}.
          </div>
          <p className="mt-1">
            Editing is disabled here to avoid losing them. Use the full
            CodeMirror editor at{" "}
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

      {editor && !disabled && <Toolbar editor={editor} />}

      <div className="overflow-hidden rounded border border-border bg-canvas">
        <EditorContent editor={editor} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void onSave()}
          disabled={saving || disabled}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:bg-stone-300"
        >
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

function Toolbar({ editor }: { editor: Editor }) {
  const btn =
    "rounded p-1.5 text-muted hover:bg-subtle hover:text-fg disabled:opacity-30 " +
    "data-[active=true]:bg-subtle data-[active=true]:text-fg";

  function setLink() {
    const prev = editor.getAttributes("link").href as string | undefined;
    const url = window.prompt("URL", prev ?? "https://");
    if (url === null) return;
    if (url === "") {
      editor.chain().focus().extendMarkRange("link").unsetLink().run();
      return;
    }
    editor.chain().focus().extendMarkRange("link").setLink({ href: url }).run();
  }

  return (
    <div className="flex flex-wrap items-center gap-1 rounded border border-border bg-panel px-2 py-1.5">
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("heading", { level: 1 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
        aria-label="H1"
        title="H1"
      >
        <Heading1 className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("heading", { level: 2 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
        aria-label="H2"
        title="H2"
      >
        <Heading2 className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("heading", { level: 3 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}
        aria-label="H3"
        title="H3"
      >
        <Heading3 className="h-4 w-4" />
      </button>
      <Divider />
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("bold")}
        onClick={() => editor.chain().focus().toggleBold().run()}
        aria-label="Bold"
        title="Bold"
      >
        <Bold className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("italic")}
        onClick={() => editor.chain().focus().toggleItalic().run()}
        aria-label="Italic"
        title="Italic"
      >
        <Italic className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("strike")}
        onClick={() => editor.chain().focus().toggleStrike().run()}
        aria-label="Strikethrough"
        title="Strikethrough"
      >
        <Strikethrough className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("code")}
        onClick={() => editor.chain().focus().toggleCode().run()}
        aria-label="Inline code"
        title="Inline code"
      >
        <Code className="h-4 w-4" />
      </button>
      <Divider />
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("bulletList")}
        onClick={() => editor.chain().focus().toggleBulletList().run()}
        aria-label="Bullet list"
        title="Bullet list"
      >
        <List className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("orderedList")}
        onClick={() => editor.chain().focus().toggleOrderedList().run()}
        aria-label="Numbered list"
        title="Numbered list"
      >
        <ListOrdered className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("blockquote")}
        onClick={() => editor.chain().focus().toggleBlockquote().run()}
        aria-label="Blockquote"
        title="Blockquote"
      >
        <Quote className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("codeBlock")}
        onClick={() => editor.chain().focus().toggleCodeBlock().run()}
        aria-label="Code block"
        title="Code block"
      >
        <code className="text-[11px] font-mono">{"{ }"}</code>
      </button>
      <Divider />
      <button
        type="button"
        className={btn}
        data-active={editor.isActive("link")}
        onClick={setLink}
        aria-label="Link"
        title="Link"
      >
        <LinkIcon className="h-4 w-4" />
      </button>
      <Divider />
      <button
        type="button"
        className={btn}
        onClick={() => editor.chain().focus().undo().run()}
        disabled={!editor.can().undo()}
        aria-label="Undo"
        title="Undo"
      >
        <Undo className="h-4 w-4" />
      </button>
      <button
        type="button"
        className={btn}
        onClick={() => editor.chain().focus().redo().run()}
        disabled={!editor.can().redo()}
        aria-label="Redo"
        title="Redo"
      >
        <Redo className="h-4 w-4" />
      </button>
    </div>
  );
}

function Divider() {
  return <span aria-hidden className="mx-1 h-4 w-px bg-border" />;
}
