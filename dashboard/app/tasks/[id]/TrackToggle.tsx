"use client";

/**
 * Two-option segmented control for a task's `track` (Experiment | Human),
 * shown in the detail-page header next to the frontmatter chips.
 *
 * Editor-gated: when `canEdit` is false it renders a static read-only
 * badge. When editable, clicking the other option POSTs to
 * `/api/tasks/track` then `router.refresh()` to re-pull the server data.
 *
 * Track is a frontmatter field, NOT a status mutation — picking a track
 * never changes the task's lifecycle column.
 */
import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import type { Track } from "@/lib/tasks";

const OPTIONS: { key: Track; label: string }[] = [
  { key: "experiment", label: "Experiment" },
  { key: "human", label: "Human" },
];

export function TrackToggle({
  taskId,
  track,
  canEdit,
}: {
  taskId: number;
  track: Track;
  canEdit: boolean;
}) {
  const router = useRouter();
  const [saving, startSaving] = useTransition();
  const [error, setError] = useState<string | null>(null);

  if (!canEdit) {
    return (
      <span className="inline-flex items-center gap-1 rounded border border-stone-200 bg-white px-2 py-0.5 text-xs">
        <span className="text-stone-500">track:</span>{" "}
        <span className="font-medium text-stone-800">
          {track === "human" ? "Human" : "Experiment"}
        </span>
      </span>
    );
  }

  function pick(next: Track) {
    if (next === track || saving) return;
    setError(null);
    startSaving(async () => {
      try {
        const res = await fetch("/api/tasks/track", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ taskId, track: next }),
        });
        const json = (await res.json()) as { ok?: boolean; error?: string };
        if (!res.ok || !json.ok) {
          setError(json.error ?? `HTTP ${res.status}`);
          return;
        }
        router.refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : "request failed");
      }
    });
  }

  return (
    <span className="inline-flex items-center gap-1.5 text-xs">
      <span className="text-stone-500">track:</span>
      <span
        className="inline-flex items-center gap-0.5 rounded-md bg-stone-100 p-0.5"
        role="group"
        aria-label="Task track"
      >
        {OPTIONS.map((o) => {
          const active = track === o.key;
          return (
            <button
              key={o.key}
              type="button"
              disabled={saving}
              aria-pressed={active}
              onClick={() => pick(o.key)}
              className={`rounded px-2 py-0.5 font-medium transition-colors disabled:opacity-50 ${
                active
                  ? o.key === "human"
                    ? "bg-fuchsia-100 text-fuchsia-800 shadow-sm"
                    : "bg-teal-100 text-teal-800 shadow-sm"
                  : "text-stone-500 hover:text-stone-800"
              }`}
            >
              {o.label}
            </button>
          );
        })}
      </span>
      {saving && <span className="text-stone-400">saving…</span>}
      {error && <span className="text-red-700">{error}</span>}
    </span>
  );
}
