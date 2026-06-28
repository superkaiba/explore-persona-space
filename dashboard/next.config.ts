import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // tasks/, docs/, logs/, updates/ all live in the repo root (sibling of
  // dashboard/). Bump the trace root one level so NFT can express
  // "include ../tasks/**" etc. without leaving the trace.
  outputFileTracingRoot: path.join(__dirname, ".."),
  // Each disk-reading route ships its data deps at runtime (the build is NOT
  // standalone, so `next start` resolves these via NFT). Keep this in lockstep
  // with what each route's lib reads:
  //   - lib/tasks.ts     -> ../tasks/**
  //   - lib/docs.ts      -> ../docs/** AND ../logs/{daily,weekly}/**
  //   - lib/logs.ts      -> ../logs/**, ../docs/**, ../tasks/**
  //   - lib/results.ts   -> ../tasks/**
  //   - lib/literature   -> ../updates/literature/**
  //   - lib/task-data.ts -> ../figures/** AND ../eval_results/** (the
  //       interactive data viewer's GET /tasks/<id>/data route, Phase 2)
  outputFileTracingIncludes: {
    // Overview: tasks (recent results) + docs (orientation docs + recent docs,
    // which include virtual dated docs under logs/).
    "/": ["../tasks/**/*", "../docs/**/*", "../logs/**/*"],
    // Tasks list + detail. The detail route renders a paper-task's committed
    // LaTeX paper (docs/papers/issue_<N>/paper.html + paper_manifest.json) when
    // `paper: true`, so trace docs/papers too.
    "/tasks": ["../tasks/**/*"],
    "/tasks/[id]": ["../tasks/**/*", "../docs/papers/**/*"],
    // Interactive data-viewer route — reads figure sidecars + the committed
    // eval_results JSON a sidecar's data_path points at.
    "/tasks/[id]/data": [
      "../tasks/**/*",
      "../figures/**/*.meta.json",
      "../figures/**/*.json",
      "../eval_results/**/*.json",
    ],
    // Cross-reference hover-preview stub for the paper render — reads the
    // target task's registry title + body frontmatter.
    "/tasks/[id]/ref": ["../tasks/**/*"],
    // Figure-serving route for the paper render — serves figures/issue_<N>/<file>
    // (the paper's relative <img> srcs are rewritten to this route).
    "/tasks/[id]/figure/[name]": ["../figures/**/*"],
    // Sessions: resolves issue numbers from the task registry to render
    // titles + links; the session-progress cache file under ~/.eps-autonomous
    // is read at request time (outside the trace root, not bundleable).
    "/sessions": ["../tasks/**/*"],
    // Results catalog + detail (public). The /results/[id] route also
    // surfaces the "Questions linked from the research hub" block, which
    // reads docs/open_questions.md — trace it.
    "/results": ["../tasks/**/*"],
    "/results/[id]": ["../tasks/**/*", "../docs/**/*"],
    // Updates feed: completed clean-results (tasks) + dated docs (docs + logs).
    "/updates": ["../tasks/**/*", "../docs/**/*", "../logs/**/*"],
    // Preview reads tasks.
    "/preview": ["../tasks/**/*"],
    // Paper render dev/smoke fixture — the sample paper + its figures.
    "/preview/paper-sample": ["../docs/papers/_sample/**/*", "../figures/**/*"],
    // Docs index + detail: docs/ plus logs/{daily,weekly} for virtual dated docs.
    "/docs": ["../docs/**/*", "../logs/**/*"],
    "/docs/[slug]": ["../docs/**/*", "../logs/**/*"],
    // Questions hub: parses docs/open_questions.md + reads the task
    // registry to decide public-vs-gated evidence links.
    "/questions": ["../docs/**/*", "../tasks/**/*"],
    // Literature reads the untouched updates/literature data dir.
    "/literature": ["../updates/literature/**/*"],
    "/literature/[date]": ["../updates/literature/**/*"],
    "/literature/papers/[id]": ["../updates/literature/**/*"],
  },
  async redirects() {
    return [
      // The /log route is retired; its feed merged into /updates.
      { source: "/log", destination: "/updates", permanent: false },
      { source: "/log/:path*", destination: "/updates", permanent: false },
      // The /docs/SUMMARY surface is retired (the auto-generated SUMMARY.md
      // was a divergent project description). docs/open_questions.md is its
      // content successor — every prior SUMMARY reader landed there for the
      // orientation prose.
      {
        source: "/docs/SUMMARY",
        destination: "/docs/open_questions",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
