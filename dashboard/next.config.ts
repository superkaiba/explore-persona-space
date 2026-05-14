import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // Vercel's File Tracing can't statically infer that we read from `../tasks/`
  // (via process.cwd()), so include the directory explicitly. The trace root
  // is bumped to the repo root so files outside `dashboard/` are reachable
  // at runtime.
  outputFileTracingRoot: path.join(__dirname, ".."),
  outputFileTracingIncludes: {
    "/": ["../tasks/**/*"],
    "/tasks/[id]": ["../tasks/**/*"],
  },
};

export default nextConfig;
