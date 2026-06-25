"use client";

/**
 * <PaperView> — renders a paper-task's LaTeX paper on /tasks/[id] (Phase C2).
 *
 * Inputs are server-prepared (lib/paper.ts): `html` is the committed paper.html,
 * RE-SANITIZED under the paperSchema and with relative figure srcs rewritten to
 * `/tasks/<N>/figure/<file>`. We mount it via dangerouslySetInnerHTML (the same
 * trusted-sanitized-HTML pattern <MarkdownDoc> uses for legacy Sagan cards) and
 * layer two interactions on top, post-mount:
 *
 *   1. Download PDF — links the manifest's pinned HF PDF URL. Disabled
 *      ("building…") when the URL is null (local / unbuilt paper).
 *   2. Cross-reference hover-preview — every `\epsref{N}`-derived
 *      `<a class="eps-ref" data-epsref="N">` gets a debounced hover card showing
 *      the target task's title + abstract (lazy-fetched from /tasks/<N>/ref,
 *      cached per id). Click opens /tasks/<N> in a new tab (the anchor's own
 *      href/target already do this; we only add the hover card). Forward-only:
 *      a target with no paper still resolves (title + excerpt).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Download, FileText, Loader2 } from "lucide-react";

type RefStub = {
  id: number;
  title: string;
  abstract: string | null;
  isPaper: boolean;
  status: string;
  exists: boolean;
};

type CardState =
  | { phase: "loading" }
  | { phase: "ready"; stub: RefStub }
  | { phase: "error" };

const HOVER_DELAY_MS = 180;
const CARD_WIDTH = 320;

export function PaperView({
  html,
  pdfUrl,
}: {
  /** Sanitized paper HTML (figure srcs already rewritten to the figure route). */
  html: string;
  /** Pinned HF PDF URL, or null → the disabled "building…" Download-PDF state. */
  pdfUrl: string | null;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  // Per-id stub cache so re-hovering the same ref doesn't refetch.
  const cacheRef = useRef<Map<number, RefStub>>(new Map());
  const hoverTimerRef = useRef<number | null>(null);
  const activeFetchRef = useRef<number | null>(null);
  const [card, setCard] = useState<
    { refId: number; top: number; left: number; state: CardState } | null
  >(null);

  const clearHoverTimer = useCallback(() => {
    if (hoverTimerRef.current != null) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
  }, []);

  const fetchStub = useCallback(async (refId: number): Promise<RefStub | null> => {
    const cached = cacheRef.current.get(refId);
    if (cached) return cached;
    try {
      const res = await fetch(`/tasks/${refId}/ref`, { credentials: "same-origin" });
      if (!res.ok) return null;
      const json = (await res.json()) as ({ ok: true } & RefStub) | { ok: false };
      if (!json.ok) return null;
      const stub: RefStub = {
        id: json.id,
        title: json.title,
        abstract: json.abstract,
        isPaper: json.isPaper,
        status: json.status,
        exists: json.exists,
      };
      cacheRef.current.set(refId, stub);
      return stub;
    } catch {
      return null;
    }
  }, []);

  // Attach hover handlers to every eps-ref anchor. Re-run when the html changes.
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const anchors = Array.from(
      root.querySelectorAll<HTMLAnchorElement>("a.eps-ref[data-epsref]"),
    );
    const cleanups: Array<() => void> = [];

    for (const a of anchors) {
      const refId = Number(a.dataset.epsref);
      if (!Number.isFinite(refId)) continue;

      const onEnter = () => {
        clearHoverTimer();
        hoverTimerRef.current = window.setTimeout(() => {
          // Position the card just below the anchor, clamped to the container.
          const aRect = a.getBoundingClientRect();
          const rootRect = root.getBoundingClientRect();
          const rawLeft = aRect.left - rootRect.left;
          const left = Math.max(
            0,
            Math.min(rawLeft, Math.max(0, root.clientWidth - CARD_WIDTH)),
          );
          const top = aRect.bottom - rootRect.top + 6;
          setCard({ refId, top, left, state: { phase: "loading" } });
          activeFetchRef.current = refId;
          void fetchStub(refId).then((stub) => {
            // Ignore a stale resolution if the user moved to another ref.
            if (activeFetchRef.current !== refId) return;
            setCard((cur) =>
              cur && cur.refId === refId
                ? {
                    ...cur,
                    state: stub
                      ? { phase: "ready", stub }
                      : { phase: "error" },
                  }
                : cur,
            );
          });
        }, HOVER_DELAY_MS);
      };

      const onLeave = () => {
        clearHoverTimer();
        // Small grace so moving onto the card itself doesn't dismiss it.
        hoverTimerRef.current = window.setTimeout(() => {
          setCard((cur) => (cur && cur.refId === refId ? null : cur));
        }, 120);
      };

      a.addEventListener("mouseenter", onEnter);
      a.addEventListener("mouseleave", onLeave);
      cleanups.push(() => {
        a.removeEventListener("mouseenter", onEnter);
        a.removeEventListener("mouseleave", onLeave);
      });
    }

    return () => {
      clearHoverTimer();
      for (const off of cleanups) off();
    };
  }, [html, clearHoverTimer, fetchStub]);

  // Keep the card alive while the pointer is over it; dismiss on leave.
  const onCardEnter = useCallback(() => clearHoverTimer(), [clearHoverTimer]);
  const onCardLeave = useCallback(() => setCard(null), []);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <PdfDownloadButton pdfUrl={pdfUrl} />
        <span className="text-xs text-stone-500">
          Rendered from the committed LaTeX paper
        </span>
      </div>

      <div ref={rootRef} className="relative">
        {/* The HTML is server-sanitized under the paperSchema (lib/paper.ts);
            mounting it directly is the same trusted-sanitized-HTML pattern
            MarkdownDoc uses for legacy Sagan cards. */}
        <div
          className="prose prose-sm sm:prose-base prose-stone max-w-none paper-html"
          dangerouslySetInnerHTML={{ __html: html }}
        />

        {card && (
          <div
            role="tooltip"
            onMouseEnter={onCardEnter}
            onMouseLeave={onCardLeave}
            style={{
              position: "absolute",
              top: card.top,
              left: card.left,
              width: CARD_WIDTH,
            }}
            className="z-30 rounded-lg border border-stone-300 bg-white p-3 text-sm shadow-lg"
          >
            <RefCard refId={card.refId} state={card.state} />
          </div>
        )}
      </div>
    </div>
  );
}

function PdfDownloadButton({ pdfUrl }: { pdfUrl: string | null }) {
  // Protocol-assert before rendering the download href: pdfUrl comes from the
  // manifest's `pdf_hf_url` and is otherwise unchecked. Require an https:// URL
  // so a `javascript:` / `data:` URL smuggled into a manifest can't become a
  // clickable href — anything else falls through to the disabled state below.
  if (!pdfUrl || !pdfUrl.startsWith("https://")) {
    return (
      <span
        className="inline-flex cursor-not-allowed items-center gap-1.5 rounded-md border border-stone-200 bg-stone-50 px-3 py-1.5 text-sm font-medium text-stone-400"
        title="The PDF is not built/uploaded yet (built --no-upload or pending)."
      >
        <Loader2 className="h-4 w-4" />
        PDF building…
      </span>
    );
  }
  return (
    <a
      href={pdfUrl}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 rounded-md border border-stone-800 bg-stone-800 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-700"
    >
      <Download className="h-4 w-4" />
      Download PDF
    </a>
  );
}

function RefCard({ refId, state }: { refId: number; state: CardState }) {
  if (state.phase === "loading") {
    return (
      <div className="flex items-center gap-2 text-xs text-stone-500">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Loading #{refId}…
      </div>
    );
  }
  if (state.phase === "error") {
    return (
      <div className="text-xs text-stone-500">
        Couldn&rsquo;t load task #{refId}.{" "}
        <a
          href={`/tasks/${refId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-stone-700 underline hover:text-stone-900"
        >
          Open ↗
        </a>
      </div>
    );
  }
  const { stub } = state;
  return (
    <a
      href={`/tasks/${refId}`}
      target="_blank"
      rel="noopener noreferrer"
      className="block no-underline"
    >
      <div className="mb-1 flex items-center gap-1.5 text-[11px] text-stone-400">
        <FileText className="h-3 w-3" />
        <span className="font-mono">#{refId}</span>
        {stub.isPaper && (
          <span className="rounded bg-amber-100 px-1 py-0.5 font-medium text-amber-800">
            paper
          </span>
        )}
        <span>· {stub.status}</span>
      </div>
      <div className="text-sm font-medium leading-snug text-stone-800">
        {stub.title}
      </div>
      {stub.abstract && (
        <p className="mt-1 line-clamp-4 text-xs leading-relaxed text-stone-600">
          {stub.abstract}
        </p>
      )}
      {!stub.exists && (
        <p className="mt-1 text-[11px] italic text-stone-400">
          (task not found)
        </p>
      )}
    </a>
  );
}
