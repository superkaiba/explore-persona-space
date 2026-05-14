export default function Literature() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold tracking-tight">Literature</h1>
      <p className="text-sm text-stone-600">
        Daily arXiv surfacing batches land here once the cron job in step 9 runs.
        Files live under <code className="rounded bg-stone-100 px-1">updates/literature/</code> in the repo.
      </p>
    </div>
  );
}
