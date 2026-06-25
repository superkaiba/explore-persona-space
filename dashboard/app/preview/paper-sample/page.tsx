/**
 * /preview/paper-sample — dev/smoke fixture for the paper render path (Phase C2).
 *
 * Renders the SAMPLE paper at docs/papers/_sample/ (built by
 * `scripts/build_paper.py --paper-dir docs/papers/_sample --jobname
 * issue_657_sample --no-upload`) through the SAME <PaperView> component the real
 * /tasks/[id] paper-task branch uses. Self-contained: it touches no real task
 * (no `paper: true` set anywhere) and reads only the committed _sample artifacts.
 *
 * Use it to eyeball the render, the Download-PDF state (the sample is built
 * --no-upload, so pdf_hf_url is null → the disabled "building…" state), the
 * cross-reference hover card (the sample's \epsref{612} resolves via
 * /tasks/612/ref), and figure serving (sample figure srcs rewrite to
 * /tasks/657/figure/<file>, which serves figures/issue_657/).
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { loadPaperFromDir } from "@/lib/paper";
import { PaperView } from "@/app/tasks/[id]/PaperView";

export const dynamic = "force-dynamic";

// The sample's logical issue number (drives the figure-route rewrite to
// figures/issue_657/ and the figure-serving route). Kept in sync with the
// build's --jobname issue_657_sample.
const SAMPLE_ISSUE = 657;

export default function PaperSamplePreview() {
  const paper = loadPaperFromDir("docs/papers/_sample", SAMPLE_ISSUE);
  if (!paper) notFound();
  return (
    <article className="space-y-6">
      <header className="space-y-2">
        <div className="flex items-baseline gap-3 text-sm text-stone-500">
          <Link href="/" className="hover:text-stone-800">
            ← Home
          </Link>
          <span>·</span>
          <span className="font-mono">paper render smoke fixture</span>
        </div>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          A sample EPS paper for dashboard render smoke
        </h1>
        <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Dev/smoke fixture — rendered from <code>docs/papers/_sample/</code>, not
          a real task. The Download-PDF button shows its disabled
          &ldquo;building&rdquo; state (the sample is built{" "}
          <code>--no-upload</code>); hover the <code>#612</code> cross-reference
          to exercise the preview card.
        </p>
      </header>
      <div className="rounded-lg border border-stone-200 bg-white p-4 sm:p-6">
        <PaperView html={paper.html} pdfUrl={paper.pdfUrl} />
      </div>
    </article>
  );
}
