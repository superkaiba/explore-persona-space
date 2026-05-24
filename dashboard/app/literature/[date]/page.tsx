import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { getDailyBatch } from "@/lib/literature";

export const dynamic = "force-dynamic";

export default async function LiteratureBatch({
  params,
}: {
  params: Promise<{ date: string }>;
}) {
  const { date } = await params;
  const batch = getDailyBatch(date);
  if (!batch) notFound();

  const fm = batch.frontmatter;
  const generatedAt = typeof fm.generated_at === "string" ? fm.generated_at : null;

  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href="/literature" className="hover:text-stone-800">
            ← All batches
          </Link>
          <span>·</span>
          <span className="font-mono">{batch.date}</span>
        </div>
        <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
          Literature — {batch.date}
        </h1>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
            <span className="text-stone-500">items:</span>{" "}
            <span className="font-medium text-stone-800">{fm.item_count ?? 0}</span>
          </span>
          <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
            <span className="text-stone-500">top score:</span>{" "}
            <span className="font-medium text-stone-800">{fm.top_score ?? 0}</span>
          </span>
          {generatedAt && (
            <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
              <span className="text-stone-500">generated:</span>{" "}
              <span className="font-medium text-stone-800">{generatedAt}</span>
            </span>
          )}
        </div>
      </header>

      <section className="prose prose-stone max-w-none sm:prose-lg">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw, rehypeHighlight]}
          components={{
            // The page header already renders the title; suppress the
            // body-level H1 to avoid duplication.
            h1: () => null,
          }}
        >
          {batch.body}
        </ReactMarkdown>
      </section>
    </article>
  );
}
