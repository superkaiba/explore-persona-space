"use client";

/**
 * Client-only wrapper around `@mdxeditor/editor`. Pulled in by
 * `CardBodyEditor.tsx` via `next/dynamic({ ssr: false })`, which is the
 * supported pattern for libraries that touch `window` at module load.
 *
 * Plugin order chosen to mirror the markdown features the clean-result
 * spec uses: headings, lists, blockquotes, thematic breaks, links, GFM
 * tables, fenced code with language pickers, plus markdown shortcuts so
 * typing `## ` still produces an H2 in the editor. `diffSourcePlugin`
 * gives a "view raw markdown" toggle in the toolbar, which we expose
 * for power users who want to drop into a textarea without leaving
 * /updates.
 *
 * `imagePlugin` is loaded with NO `imageUploadHandler` — pasted/inserted
 * image URLs round-trip as `![alt](url)`, but the toolbar's
 * `InsertImage` button is intentionally omitted from the toolbar config
 * (non-goal per the brief: no image uploading).
 */

import {
  MDXEditor,
  type MDXEditorMethods,
  type MDXEditorProps,
  headingsPlugin,
  listsPlugin,
  quotePlugin,
  thematicBreakPlugin,
  markdownShortcutPlugin,
  linkPlugin,
  linkDialogPlugin,
  imagePlugin,
  tablePlugin,
  codeBlockPlugin,
  codeMirrorPlugin,
  diffSourcePlugin,
  toolbarPlugin,
  UndoRedo,
  BoldItalicUnderlineToggles,
  BlockTypeSelect,
  CreateLink,
  InsertTable,
  InsertCodeBlock,
  InsertThematicBreak,
  ListsToggle,
  DiffSourceToggleWrapper,
  Separator,
} from "@mdxeditor/editor";
import type { RefObject } from "react";

import "@mdxeditor/editor/style.css";
import "./CardBodyEditor.css";

// Code-block language picker contents. Keys are the fence-info tokens
// MDXEditor will write back; values are the labels shown in the
// dropdown. The empty-string key covers fenced blocks without a
// language (very common in /updates bodies: the Reproducibility
// `git clone` block, shell snippets, etc.).
const CODE_BLOCK_LANGUAGES: Record<string, string> = {
  "": "Plain text",
  bash: "Bash",
  sh: "Shell",
  python: "Python",
  py: "Python",
  ts: "TypeScript",
  tsx: "TSX",
  js: "JavaScript",
  jsx: "JSX",
  json: "JSON",
  yaml: "YAML",
  md: "Markdown",
  sql: "SQL",
};

export default function CardBodyEditorClient({
  editorRef,
  initialMarkdown,
  onError,
}: {
  editorRef: RefObject<MDXEditorMethods | null>;
  initialMarkdown: string;
  onError?: MDXEditorProps["onError"];
}) {
  return (
    <div className="overflow-hidden rounded border border-border bg-canvas">
      <MDXEditor
        ref={editorRef}
        markdown={initialMarkdown}
        onError={onError}
        contentEditableClassName={
          "prose prose-sm max-w-none focus:outline-none min-h-[280px] px-4 py-3 " +
          "prose-headings:text-fg prose-p:text-fg-soft prose-strong:text-fg " +
          "prose-code:text-fg prose-pre:border prose-pre:border-border " +
          "prose-pre:bg-subtle prose-li:text-fg-soft prose-a:text-accent"
        }
        plugins={[
          headingsPlugin(),
          listsPlugin(),
          quotePlugin(),
          thematicBreakPlugin(),
          linkPlugin(),
          linkDialogPlugin(),
          imagePlugin(),
          tablePlugin(),
          // codeBlockPlugin MUST be loaded BEFORE codeMirrorPlugin —
          // codeMirrorPlugin registers itself as a descriptor that
          // matches the languages we list, and code blocks whose lang
          // does not match fall back to the codeBlockPlugin default.
          codeBlockPlugin({ defaultCodeBlockLanguage: "" }),
          codeMirrorPlugin({ codeBlockLanguages: CODE_BLOCK_LANGUAGES }),
          diffSourcePlugin({
            viewMode: "rich-text",
            diffMarkdown: initialMarkdown,
          }),
          markdownShortcutPlugin(),
          toolbarPlugin({
            toolbarClassName: "epm-mdxeditor-toolbar",
            toolbarContents: () => (
              <DiffSourceToggleWrapper options={["rich-text", "source"]}>
                <UndoRedo />
                <Separator />
                <BoldItalicUnderlineToggles />
                <Separator />
                <BlockTypeSelect />
                <Separator />
                <ListsToggle />
                <Separator />
                <CreateLink />
                <InsertTable />
                <InsertCodeBlock />
                <InsertThematicBreak />
              </DiffSourceToggleWrapper>
            ),
          }),
        ]}
      />
    </div>
  );
}
