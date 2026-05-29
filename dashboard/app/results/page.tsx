/**
 * /results — the public clean-result catalog.
 *
 * Public surface (no auth gate; the proxy allowlists `/results`). The data
 * layer (@/lib/results#listPublicResults) applies the AUTHORITATIVE predicate:
 * a task is a result iff it is `completed` AND its `classification`
 * frontmatter field is exactly "useful" (NOT a prose regex), excluding
 * `format-exemplar`-tagged tasks. Cards link to /results/[id]; filtering
 * (confidence/topic/date/search) happens client-side over the pre-loaded,
 * public-safe listing.
 */
import { listPublicResults, publicResultTags } from "@/lib/results";
import { ResultsBrowser } from "./ResultsBrowser";

export const dynamic = "force-dynamic";

export default async function ResultsPage() {
  const results = listPublicResults();
  const allTags = publicResultTags(results);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Results</h1>
        <p className="mt-1 max-w-2xl text-sm text-stone-600">
          Promoted clean results from the Explore Persona Space project — completed
          experiments whose findings were classified useful. Each links to a
          self-contained write-up with its confidence level, methodology, and figures.
        </p>
      </header>

      <ResultsBrowser results={results} allTags={allTags} />
    </div>
  );
}
