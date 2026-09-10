import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  /* config options here */
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  // duckdb uses native bindings (node-pre-gyp) that can't be bundled by webpack
  serverExternalPackages: ["duckdb"],
};

export default nextConfig;
