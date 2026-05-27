"use client";

/**
 * Client wrapper inside the BodyCard. Default state renders the markdown
 * children that the server passed in (already-rendered React tree, so the
 * server-side ReactMarkdown + rehype-raw + rehype-highlight pipeline stays
 * authoritative). Clicking Edit swaps to `InlineBodyEditor`.
 *
 * Why hand the rendered markdown tree in as `children` rather than the
 * raw body string + a client-side ReactMarkdown call:
 *   - Keeps server-side rendering of the body — first paint matches the
 *     /tasks/[id] route output before this commit, no hydration penalty.
 *   - Avoids re-shipping `react-markdown` + plugin bundles to clients
 *     just to render a card that's read-only 99% of the time.
 * When the user is mid-edit we hide the rendered tree and show the
 * editor; on Save we keep showing the (now stale) tree until
 * `router.refresh()` returns fresh server HTML for the new body.
 */
import { Pencil } from "lucide-react";
import { useState } from "react";
import { InlineBodyEditor } from "./InlineBodyEditor";

export function EditableBody({
  taskId,
  initialBody,
  canEdit,
  children,
}: {
  taskId: number;
  initialBody: string;
  canEdit: boolean;
  children: React.ReactNode;
}) {
  const [editing, setEditing] = useState(false);

  if (!canEdit) {
    return <>{children}</>;
  }

  return (
    <div className="space-y-3">
      {!editing && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="inline-flex items-center gap-1.5 rounded border border-stone-300 bg-white px-2.5 py-1 text-xs text-stone-700 hover:bg-stone-50"
          >
            <Pencil className="h-3 w-3" />
            Edit body
          </button>
        </div>
      )}
      {editing ? (
        <InlineBodyEditor
          taskId={taskId}
          initialBody={initialBody}
          onSaved={() => setEditing(false)}
          onCancel={() => setEditing(false)}
        />
      ) : (
        children
      )}
    </div>
  );
}
