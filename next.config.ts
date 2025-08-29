// @ts-ignore
const nextConfig: any = {
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
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME || "Accessibility-First Call Center System - POC",
    NEXT_PUBLIC_APP_VERSION: process.env.NEXT_PUBLIC_APP_VERSION || "POC v1.0",
    NEXT_PUBLIC_DEMO_MODE: process.env.NEXT_PUBLIC_DEMO_MODE || "true",
  },
  // ปิด ESLint สำหรับ POC - สำคัญมาก!
  eslint: {
    ignoreDuringBuilds: true,
  },
  // ปิด TypeScript type checking สำหรับ POC - สำคัญมาก!
  typescript: {
    ignoreBuildErrors: true,
  },
  // ปิด telemetry
  telemetry: false,
};

export default nextConfig; 