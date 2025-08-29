import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone',
  experimental: {
    turbo: {
      rules: {
        '*.svg': {
          loaders: ['@svgr/webpack'],
          as: '*.js',
        },
      },
    },
  },
  images: {
    domains: ['localhost', 'your-backend-url.com'],
    unoptimized: true,
  },
  env: {
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME || "Accessibility-First Call Center System",
    NEXT_PUBLIC_APP_VERSION: process.env.NEXT_PUBLIC_APP_VERSION || "1.0.0",
    NEXT_PUBLIC_DEMO_MODE: process.env.NEXT_PUBLIC_DEMO_MODE || "true",
  },
  // ปิด ESLint สำหรับ POC
  eslint: {
    ignoreDuringBuilds: true,
  },
  // ปิด TypeScript type checking สำหรับ POC
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
