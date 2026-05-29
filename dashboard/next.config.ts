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
  //   - lib/tasks.ts   -> ../tasks/**
  //   - lib/docs.ts    -> ../docs/** AND ../logs/{daily,weekly}/**
  //   - lib/logs.ts    -> ../logs/**, ../docs/**, ../tasks/**
  //   - lib/results.ts -> ../tasks/**
  //   - lib/literature -> ../updates/literature/**
  outputFileTracingIncludes: {
    // Overview: tasks (recent results) + docs (orientation docs + recent docs,
    // which include virtual dated docs under logs/).
    "/": ["../tasks/**/*", "../docs/**/*", "../logs/**/*"],
    // Tasks list + detail.
    "/tasks": ["../tasks/**/*"],
    "/tasks/[id]": ["../tasks/**/*"],
    // Results catalog + detail (public).
    "/results": ["../tasks/**/*"],
    "/results/[id]": ["../tasks/**/*"],
    // Updates feed: completed clean-results (tasks) + dated docs (docs + logs).
    "/updates": ["../tasks/**/*", "../docs/**/*", "../logs/**/*"],
    // Preview reads tasks.
    "/preview": ["../tasks/**/*"],
    // Docs index + detail: docs/ plus logs/{daily,weekly} for virtual dated docs.
    "/docs": ["../docs/**/*", "../logs/**/*"],
    "/docs/[slug]": ["../docs/**/*", "../logs/**/*"],
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
    ];
  },
};

export default nextConfig;
