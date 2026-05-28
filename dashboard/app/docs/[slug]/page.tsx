import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import path from "node:path";
import { getDoc } from "@/lib/docs";

export const dynamic = "force-dynamic";

const REPO_BLOB = "https://github.com/superkaiba/explore-persona-space/blob/main";

// Cross-doc links in the markdown are repo-relative paths. Rewrite them so they
// resolve in the browser:
//   - flat same-dir doc (`./open_questions.md`, `papers.md#sec`) -> `/docs/<slug>`
//   - any other repo-relative `.md` (`../RESULTS.md`, `sub/dir.md`) -> the public
//     GitHub source, resolved relative to docs/ (the /docs route only serves
//     top-level docs/*.md, so these can't be internal links)
// External URLs, mailto, absolute paths, and pure anchors pass through untouched;
// external links open in a new tab (see the `a` component below).
function rewriteHref(href: string | undefined): string | undefined {
  if (!href) return href;
  if (/^(https?:|mailto:|#|\/)/.test(href)) return href;
  const flat = href.match(/^(?:\.\/)?([A-Za-z0-9._-]+)\.md(#.*)?$/);
  if (flat) return `/docs/${flat[1]}${flat[2] ?? ""}`;
  const md = href.match(/^([^#]+\.md)(#.*)?$/);
  if (md) {
    const resolved = path.posix.normalize(path.posix.join("docs", md[1]));
    if (resolved.startsWith("..")) return href; // escapes repo root; leave as-is
    return `${REPO_BLOB}/${resolved}${md[2] ?? ""}`;
  }
  return href;
}

export default async function DocPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const doc = getDoc(slug);
  if (!doc) notFound();

  const status = typeof doc.frontmatter.status === "string" ? doc.frontmatter.status : null;

  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href="/docs" className="hover:text-stone-800">
            ← All docs
          </Link>
          <span>·</span>
          <span className="font-mono">{doc.slug}.md</span>
        </div>
        <h1 className="text-2xl font-semibold leading-snug tracking-tight sm:text-3xl">
          {doc.title}
        </h1>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {status && (
            <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
              <span className="text-stone-500">status:</span>{" "}
              <span className="font-medium text-stone-800">{status}</span>
            </span>
          )}
          {doc.lastUpdated && (
            <span className="rounded border border-stone-200 bg-white px-2 py-0.5">
              <span className="text-stone-500">updated:</span>{" "}
              <span className="font-medium text-stone-800">{doc.lastUpdated}</span>
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
            a: ({ href, children }) => {
              const rewritten = rewriteHref(href);
              const external = !!rewritten && /^https?:\/\//.test(rewritten);
              return (
                <a
                  href={rewritten}
                  {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
                >
                  {children}
                </a>
              );
            },
          }}
        >
          {doc.body}
        </ReactMarkdown>
      </section>
    </article>
  );
}
