import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // tasks/ lives in the repo root (sibling of dashboard/). Bump the trace
  // root one level so NFT can express "include ../tasks/**" without
  // leaving the trace.
  outputFileTracingRoot: path.join(__dirname, ".."),
  outputFileTracingIncludes: {
    "/": ["../tasks/**/*"],
    "/tasks/[id]": ["../tasks/**/*"],
    "/literature": ["../updates/literature/**/*"],
    "/literature/[date]": ["../updates/literature/**/*"],
    "/literature/papers/[id]": ["../updates/literature/**/*"],
  },
};

export default nextConfig;
