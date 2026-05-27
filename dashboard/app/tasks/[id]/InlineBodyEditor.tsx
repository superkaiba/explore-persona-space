"use client";

/**
 * Compact in-place body editor for the task detail page (`/tasks/[id]`).
 *
 * Mirrors the standalone `/tasks/[id]/edit` route — same CodeMirror
 * widget, same Ctrl/Cmd-S, same verify_task_body.py button — but lives
 * inside the BodyCard so a logged-in editor can swap to edit mode
 * without leaving the timeline view. Save flow shells out to
 * `saveTaskBody` (which calls `uv run python scripts/task.py set-body`)
 * via the existing server action. No new API route needed.
 *
 * On Save success: bubble the saved markdown up via `onSaved` so the
 * parent immediately reflects the change, AND call `router.refresh()`
 * to pull fresh server data (frontmatter chips, status pill, registry
 * cache) without a full reload.
 */
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState, useTransition } from "react";
import { markdown } from "@codemirror/lang-markdown";
import { saveTaskBody, verifyTaskBody } from "./edit/actions";

const CodeMirror = dynamic(
  () => import("@uiw/react-codemirror").then((m) => m.default),
  {
    ssr: false,
    loading: () => (
      <div className="h-[50vh] animate-pulse rounded bg-stone-100" aria-hidden />
    ),
  },
);

const EXTENSIONS = [markdown()];

export function InlineBodyEditor({
  taskId,
  initialBody,
  onSaved,
  onCancel,
}: {
  taskId: number;
  initialBody: string;
  onSaved: (newBody: string) => void;
  onCancel: () => void;
}) {
  const router = useRouter();
  const [body, setBody] = useState(initialBody);
  const [savedBody, setSavedBody] = useState(initialBody);
  const [saving, startSaving] = useTransition();
  const [verifying, startVerifying] = useTransition();
  const [saveError, setSaveError] = useState<string | null>(null);
  const [verifyOk, setVerifyOk] = useState<boolean | null>(null);
  const [verifyOutput, setVerifyOutput] = useState<string | null>(null);

  const isDirty = body !== savedBody;

  const onChange = useCallback((val: string) => {
    setBody(val);
    setSaveError(null);
  }, []);

  const onSave = useCallback(() => {
    if (!isDirty || saving) return;
    const snapshot = body;
    setSaveError(null);
    startSaving(async () => {
      const res = await saveTaskBody(taskId, snapshot);
      if (res.ok) {
        setSavedBody(snapshot);
        onSaved(snapshot);
        router.refresh();
      } else {
        setSaveError(res.error);
      }
    });
  }, [body, isDirty, saving, taskId, onSaved, router]);

  // Ctrl/Cmd-S → save. The dependency on `onSave` keeps the latest
  // closure in scope; without it the listener would call a stale
  // version that captured the initial `body`.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        onSave();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onSave]);

  const onVerify = useCallback(() => {
    if (verifying) return;
    startVerifying(async () => {
      const res = await verifyTaskBody(body);
      setVerifyOk(res.ok);
      setVerifyOutput(res.output || "(no output)");
    });
  }, [body, verifying]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onSave}
          disabled={!isDirty || saving}
          className="rounded bg-stone-900 px-3 py-1.5 text-sm font-medium text-white disabled:bg-stone-300"
        >
          {saving ? "Saving…" : isDirty ? "Save (⌘/Ctrl-S)" : "Saved"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={saving}
          className="rounded border border-stone-300 bg-white px-3 py-1.5 text-sm font-medium text-stone-800 hover:bg-stone-50 disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onVerify}
          disabled={verifying}
          className="rounded border border-stone-300 bg-white px-3 py-1.5 text-sm font-medium text-stone-800 hover:bg-stone-50 disabled:opacity-50"
        >
          {verifying ? "Verifying…" : "Run verify_task_body.py"}
        </button>
        {isDirty && !saving && !saveError && (
          <span className="text-xs text-amber-700">Unsaved changes</span>
        )}
        {saveError && <span className="text-xs text-red-700">{saveError}</span>}
      </div>

      <div className="overflow-hidden rounded border border-stone-300 bg-white">
        <CodeMirror
          value={body}
          extensions={EXTENSIONS}
          onChange={onChange}
          height="50vh"
          basicSetup={{
            lineNumbers: true,
            highlightActiveLine: true,
            highlightActiveLineGutter: true,
            foldGutter: true,
            indentOnInput: true,
          }}
        />
      </div>

      {verifyOutput !== null && (
        <div
          className={`rounded border px-3 py-2 text-xs ${
            verifyOk
              ? "border-emerald-300 bg-emerald-50 text-emerald-900"
              : "border-red-300 bg-red-50 text-red-900"
          }`}
        >
          <div className="mb-1 font-medium">
            {verifyOk
              ? "verify_task_body.py PASS"
              : "verify_task_body.py FAIL"}
          </div>
          <pre className="overflow-auto whitespace-pre-wrap font-mono leading-relaxed">
            {verifyOutput}
          </pre>
        </div>
      )}
    </div>
  );
}
