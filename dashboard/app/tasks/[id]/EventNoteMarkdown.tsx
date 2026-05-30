"use client";

/**
 * Client-rendered markdown for an event-feed note.
 *
 * Why a client component: the task page is a Server Component, and RSC
 * renders the *children* of client components (CollapsiblePanel) on the
 * server — so an inline `<ReactMarkdown>` note ran server-side for EVERY
 * event card on every request, even collapsed ones. On tasks with a large
 * events.jsonl (e.g. #377, ~450 KB) that dominated the server render
 * (~1.5 s TTFB). Moving the markdown behind a "use client" boundary means
 * the server only serializes the raw note string; the markdown parse runs
 * in the browser, and (because EventCard is collapsed by default) only
 * when the user actually expands a card.
 */
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";

export function EventNoteMarkdown({ note }: { note: string }) {
  return (
    <div className="prose prose-sm prose-stone max-w-none sm:prose-base">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, rehypeHighlight]}>
        {note}
      </ReactMarkdown>
    </div>
  );
}
