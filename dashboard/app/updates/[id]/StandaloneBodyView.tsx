"use client";

import { useState } from "react";
import { Loader2, Pencil, Sparkles } from "lucide-react";
import { CardBodyEditor } from "@/components/updates/CardBodyEditor";
import { CardCommentBox } from "@/components/updates/CardCommentBox";

/**
 * Client wrapper for the standalone /updates/[id] page. Mirrors the
 * modal full-view's Edit + Address-comments affordances, but in a
 * page-shaped container.
 *
 * State:
 *   - editing: when true, body swaps from CardCommentBox → CardBodyEditor
 *   - editedMarkdown: holds Save'd body so we reflect the change without
 *     a hard reload
 *   - addressing: pending state on the Address-comments POST
 *   - unaddressedCount: lifted from CardCommentBox so the button can
 *     disable when there are no open comments to address
 *   - refreshNonce: bumping it forces CardCommentBox to re-fetch comments
 *     (used after Address-comments rewrites the file)
 */
export function StandaloneBodyView({
  taskId,
  initialMarkdown,
  currentUserEmail,
  canEdit,
}: {
  taskId: number;
  initialMarkdown: string;
  currentUserEmail: string | null;
  canEdit: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [editedMarkdown, setEditedMarkdown] = useState<string | null>(null);
  const [addressing, setAddressing] = useState(false);
  const [addressError, setAddressError] = useState<string | null>(null);
  const [unaddressedCount, setUnaddressedCount] = useState(0);
  const [refreshNonce, setRefreshNonce] = useState(0);

  const markdown = editedMarkdown ?? initialMarkdown;

  async function onAddressComments() {
    if (!canEdit || addressing) return;
    setAddressing(true);
    setAddressError(null);
    try {
      const res = await fetch("/api/updates/address-comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ taskId }),
      });
      if (!res.ok) {
        const j = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(j.error ?? `address-comments failed: ${res.status}`);
      }
      const j = (await res.json()) as { ok: true; body?: string };
      if (typeof j.body === "string") setEditedMarkdown(j.body);
      setRefreshNonce((n) => n + 1);
    } catch (e) {
      setAddressError(e instanceof Error ? e.message : String(e));
    } finally {
      setAddressing(false);
    }
  }

  return (
    <>
      {canEdit && (
        <div className="mb-6 flex flex-wrap items-center gap-2">
          {!editing && (
            <button
              type="button"
              onClick={() => setEditing(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-panel px-2.5 py-1.5 text-[12px] text-fg hover:bg-subtle"
              title="Edit this result body"
            >
              <Pencil className="h-3.5 w-3.5" />
              Edit
            </button>
          )}
          {!editing && (
            <button
              type="button"
              onClick={() => void onAddressComments()}
              disabled={addressing || unaddressedCount === 0}
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-panel px-2.5 py-1.5 text-[12px] text-fg hover:bg-subtle disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-panel"
              title={
                unaddressedCount === 0
                  ? "No open comments to address"
                  : `Address ${unaddressedCount} open comment${unaddressedCount === 1 ? "" : "s"} with Claude`
              }
            >
              {addressing ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Sparkles className="h-3.5 w-3.5" />
              )}
              Address comments
              {unaddressedCount > 0 && (
                <span className="ml-1 rounded bg-subtle px-1 text-[10px] tabular-nums">
                  {unaddressedCount}
                </span>
              )}
            </button>
          )}
        </div>
      )}

      {addressError && (
        <div className="mb-4 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-[12px] text-amber-900">
          Address-comments failed: {addressError}
        </div>
      )}

      {editing && canEdit ? (
        <CardBodyEditor
          taskId={taskId}
          initialMarkdown={markdown}
          onSaved={(md) => {
            setEditedMarkdown(md);
            setEditing(false);
          }}
          onCancel={() => setEditing(false)}
        />
      ) : (
        <CardCommentBox
          taskId={taskId}
          body={markdown}
          currentUserEmail={currentUserEmail}
          layout="rail"
          onUnaddressedChange={setUnaddressedCount}
          refreshNonce={refreshNonce}
        />
      )}
    </>
  );
}
