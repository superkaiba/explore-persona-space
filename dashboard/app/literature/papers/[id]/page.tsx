import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import { getPaper, type PaperFrontmatter } from "@/lib/literature";

export const dynamic = "force-dynamic";

export default async function LiteraturePaper({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const paper = getPaper(id);
  if (!paper) notFound();

  const fm = paper.frontmatter;

  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href="/literature" className="hover:text-stone-800">
            ← All batches
          </Link>
          <span>·</span>
          <span className="font-mono">{paper.slug}</span>
        </div>
        <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
          {fm.title || paper.slug}
        </h1>
        <PaperMeta fm={fm} />
      </header>

      <section className="prose prose-stone max-w-none sm:prose-lg">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw, rehypeHighlight]}
          components={{
            // The page header renders the title already; suppress body H1.
            h1: () => null,
          }}
        >
          {paper.body}
        </ReactMarkdown>
      </section>
    </article>
  );
}

function PaperMeta({ fm }: { fm: PaperFrontmatter }) {
  const surfacedDays = Array.isArray(fm.surfaced_days) ? fm.surfaced_days : [];
  const categories = Array.isArray(fm.categories) ? fm.categories : [];

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {fm.topic && (
        <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
          <span className="text-stone-500">topic:</span>{" "}
          <span className="font-medium text-stone-800">{fm.topic}</span>
        </span>
      )}
      {typeof fm.highest_score === "number" && (
        <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
          <span className="text-stone-500">top score:</span>{" "}
          <span className="font-medium text-stone-800">{fm.highest_score}</span>
        </span>
      )}
      {fm.released_on && (
        <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
          <span className="text-stone-500">released:</span>{" "}
          <span className="font-medium text-stone-800">{fm.released_on}</span>
        </span>
      )}
      {fm.first_surfaced_on && (
        <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
          <span className="text-stone-500">first surfaced:</span>{" "}
          <span className="font-medium text-stone-800">{fm.first_surfaced_on}</span>
        </span>
      )}
      {fm.url && (
        <a
          href={fm.url}
          target="_blank"
          rel="noreferrer"
          className="rounded bg-stone-900 px-2 py-0.5 text-white hover:bg-stone-700"
        >
          arXiv
        </a>
      )}
      {fm.pdf_url && (
        <a
          href={fm.pdf_url}
          target="_blank"
          rel="noreferrer"
          className="rounded bg-stone-100 px-2 py-0.5 text-stone-700 hover:bg-stone-200"
        >
          PDF
        </a>
      )}
      {categories.map((c) => (
        <span
          key={`cat-${c}`}
          className="rounded bg-amber-50 px-2 py-0.5 text-amber-800"
        >
          {c}
        </span>
      ))}
      {surfacedDays.map((d) => (
        <Link
          key={`day-${d}`}
          href={`/literature/${d}`}
          className="rounded bg-stone-100 px-2 py-0.5 text-stone-700 hover:bg-stone-200"
        >
          {d}
        </Link>
      ))}
    </div>
  );
}
