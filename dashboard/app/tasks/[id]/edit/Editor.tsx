"use client";

import { useCallback, useEffect, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { markdown } from "@codemirror/lang-markdown";
import { saveTaskBody, verifyTaskBody } from "./actions";

// CodeMirror touches `window` during module init; load only on the client.
const CodeMirror = dynamic(
  () => import("@uiw/react-codemirror").then((m) => m.default),
  { ssr: false, loading: () => <EditorSkeleton /> },
);

const EXTENSIONS = [markdown()];

export function Editor({ taskId, initialBody }: { taskId: number; initialBody: string }) {
  const [body, setBody] = useState(initialBody);
  const [savedBody, setSavedBody] = useState(initialBody);
  const [verifyOutput, setVerifyOutput] = useState<string | null>(null);
  const [verifyOk, setVerifyOk] = useState<boolean | null>(null);
  const [saveMessage, setSaveMessage] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [saving, startSaving] = useTransition();
  const [verifying, startVerifying] = useTransition();
  const router = useRouter();

  const isDirty = body !== savedBody;

  const onChange = useCallback((val: string) => {
    setBody(val);
    setSaveMessage(null);
  }, []);

  // Ctrl/Cmd-S → save.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        onSave();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [body, isDirty]);

  // Warn before navigating away with unsaved changes.
  useEffect(() => {
    function onBeforeUnload(e: BeforeUnloadEvent) {
      if (!isDirty) return;
      e.preventDefault();
      e.returnValue = "";
    }
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [isDirty]);

  function onSave() {
    if (!isDirty || saving) return;
    const snapshot = body;
    startSaving(async () => {
      const result = await saveTaskBody(taskId, snapshot);
      if (result.ok) {
        setSavedBody(snapshot);
        setSaveMessage({ kind: "ok", text: "Saved." });
        // Pull fresh server data so the chrome (status pill, frontmatter)
        // reflects any side effects.
        router.refresh();
      } else {
        setSaveMessage({ kind: "err", text: result.error });
      }
    });
  }

  function onVerify() {
    if (verifying) return;
    startVerifying(async () => {
      const result = await verifyTaskBody(body);
      setVerifyOk(result.ok);
      setVerifyOutput(result.output || "(no output)");
    });
  }

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
          onClick={onVerify}
          disabled={verifying}
          className="rounded border border-stone-300 bg-white px-3 py-1.5 text-sm font-medium text-stone-800 disabled:opacity-50"
        >
          {verifying ? "Verifying…" : "Run verify_task_body.py"}
        </button>
        {saveMessage && (
          <span
            className={
              saveMessage.kind === "ok"
                ? "text-sm text-emerald-700"
                : "text-sm text-red-700"
            }
          >
            {saveMessage.text}
          </span>
        )}
        {isDirty && !saving && !saveMessage && (
          <span className="text-xs text-amber-700">Unsaved changes</span>
        )}
      </div>

      <div className="overflow-hidden rounded border border-stone-300 bg-white">
        <CodeMirror
          value={body}
          extensions={EXTENSIONS}
          onChange={onChange}
          height="60vh"
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
            {verifyOk ? "verify_task_body.py PASS" : "verify_task_body.py FAIL"}
          </div>
          <pre className="overflow-auto whitespace-pre-wrap font-mono leading-relaxed">
            {verifyOutput}
          </pre>
        </div>
      )}
    </div>
  );
}

function EditorSkeleton() {
  return (
    <div className="h-[60vh] animate-pulse rounded bg-stone-100" aria-hidden />
  );
}
