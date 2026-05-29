import Link from "next/link";
import { notFound } from "next/navigation";
import path from "node:path";
import { getDoc } from "@/lib/docs";
import { commentsPathForSlug, readComments } from "@/lib/doc-comments";
import { isEditorAuthed, requireSessionAuth } from "@/lib/auth";
import { DocBody, type DocCommentView } from "./DocBody";

export const dynamic = "force-dynamic";

const REPO_BLOB = "https://github.com/superkaiba/explore-persona-space/blob/main";

// Cross-doc links in the markdown are repo-relative paths. Rewrite them so they
// resolve in the browser BEFORE handing the body to <MarkdownDoc> (which has no
// per-link transform hook). Two cases:
//   - flat same-dir doc (`./open_questions.md`, `papers.md#sec`) -> `/docs/<slug>`
//   - any other repo-relative `.md` (`../RESULTS.md`, `sub/dir.md`) -> the public
//     GitHub source, resolved relative to docs/ (the /docs route serves only
//     top-level docs/*.md + the virtual stores, so these can't be internal).
// External URLs, mailto, absolute paths, and pure anchors pass through untouched.
function rewriteHref(href: string): string {
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

// Rewrite inline-link + reference-definition targets in the markdown source.
// Only touches the URL token of `[text](url)` and `[id]: url`; leaves code
// spans / fenced blocks alone because the patterns require markdown link
// syntax. Conservative: a malformed match is left untouched.
function rewriteMarkdownLinks(md: string): string {
  // Inline links: [text](url) or [text](url "title")
  let out = md.replace(
    /(\]\()([^)\s]+)((?:\s+"[^"]*")?\))/g,
    (_m, open: string, url: string, close: string) => `${open}${rewriteHref(url)}${close}`,
  );
  // Reference definitions: [id]: url  (start-of-line)
  out = out.replace(
    /^(\s{0,3}\[[^\]]+\]:\s+)(\S+)/gm,
    (_m, prefix: string, url: string) => `${prefix}${rewriteHref(url)}`,
  );
  return out;
}

// Strip a single leading top-level H1 so the doc title isn't shown twice (the
// page header already renders it). Mirrors the old `h1: () => null` override.
function stripLeadingH1(body: string): string {
  return body.replace(/^\s*#\s+.+?\r?\n/, "");
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

  // Initial comments (server-read; the client refreshes via /api/docs/comment).
  const commentsFile = commentsPathForSlug(slug);
  const rawComments = commentsFile ? await readComments(commentsFile) : [];
  const initialComments: DocCommentView[] = rawComments.map((c) => ({
    id: c.id,
    ts: c.ts,
    author: c.author,
    kind: c.kind,
    body: c.body,
    section_label: c.section_label,
    quote: c.quote,
    in_reply_to: c.in_reply_to,
    addressed: c.addressed,
    addressed_note: c.addressed_note,
  }));

  const user = await requireSessionAuth();
  const editorAuthed = await isEditorAuthed();

  const renderedBody = rewriteMarkdownLinks(stripLeadingH1(doc.body));

  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <div className="flex flex-wrap items-baseline gap-3 text-sm text-stone-500">
          <Link href="/docs" className="hover:text-stone-800">
            ← All docs
          </Link>
          <span>·</span>
          <span className="rounded bg-stone-100 px-1.5 py-0.5 text-xs font-medium text-stone-600">
            {doc.category}
          </span>
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

      <DocBody
        slug={slug}
        body={renderedBody}
        title={doc.title}
        initialComments={initialComments}
        editorAuthed={editorAuthed}
        currentUserEmail={user?.email ?? null}
      />
    </article>
  );
}
