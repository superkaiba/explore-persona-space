"use client";

/**
 * In-place editor for the task title (shown in the page header). Authed
 * users see a Pencil button next to the H1; clicking swaps the heading
 * for an input + Save/Cancel pair. Save shells out to
 * `saveTaskTitle` → `uv run python scripts/task.py set-title <N> "..."`.
 *
 * Body markdown frontmatter is NOT edited here — we intentionally only
 * surface `title`. Read-only / workflow-managed fields (`created_at`,
 * `parent_id`, `goal`, `has_clean_result`, ...) stay under the
 * task.py-set helpers that own them.
 */
import { Pencil } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, useTransition } from "react";
import { saveTaskTitle } from "./edit/actions";

export function TitleEditor({
  taskId,
  initialTitle,
  canEdit,
}: {
  taskId: number;
  initialTitle: string;
  canEdit: boolean;
}) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(initialTitle);
  const [saving, startSaving] = useTransition();
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // initialTitle changes after router.refresh() — keep the draft in sync
  // when we're NOT in edit mode (don't trample user typing). Sync-from-
  // prop is the textbook setState-in-effect case React allows.
  useEffect(() => {
    if (!editing) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDraft(initialTitle);
    }
  }, [initialTitle, editing]);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  function onCancel() {
    setDraft(initialTitle);
    setEditing(false);
    setError(null);
  }

  function onSave() {
    const trimmed = draft.trim();
    if (!trimmed || trimmed === initialTitle.trim()) {
      setEditing(false);
      return;
    }
    setError(null);
    startSaving(async () => {
      const res = await saveTaskTitle(taskId, trimmed);
      if (res.ok) {
        setEditing(false);
        router.refresh();
      } else {
        setError(res.error);
      }
    });
  }

  if (!canEdit) {
    return (
      <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
        {initialTitle || "(untitled)"}
      </h1>
    );
  }

  if (editing) {
    return (
      <div className="space-y-2">
        <input
          ref={inputRef}
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              onSave();
            } else if (e.key === "Escape") {
              e.preventDefault();
              onCancel();
            }
          }}
          disabled={saving}
          className="w-full rounded border border-stone-300 bg-white px-2 py-1 text-2xl font-semibold leading-snug tracking-tight focus:border-stone-500 focus:outline-none sm:text-3xl"
        />
        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            onClick={onSave}
            disabled={saving}
            className="rounded bg-stone-900 px-2.5 py-1 font-medium text-white disabled:bg-stone-300"
          >
            {saving ? "Saving…" : "Save (Enter)"}
          </button>
          <button
            type="button"
            onClick={onCancel}
            disabled={saving}
            className="rounded border border-stone-300 bg-white px-2.5 py-1 font-medium text-stone-800 hover:bg-stone-50 disabled:opacity-50"
          >
            Cancel (Esc)
          </button>
          {error && <span className="text-red-700">{error}</span>}
        </div>
      </div>
    );
  }

  return (
    <div className="group flex items-baseline gap-2">
      <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
        {initialTitle || "(untitled)"}
      </h1>
      <button
        type="button"
        onClick={() => setEditing(true)}
        title="Edit title"
        aria-label="Edit title"
        className="rounded p-1 text-stone-400 opacity-0 transition-opacity hover:bg-stone-100 hover:text-stone-700 group-hover:opacity-100 focus:opacity-100"
      >
        <Pencil className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
