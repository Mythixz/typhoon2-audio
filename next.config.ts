// @ts-ignore
const nextConfig: any = {
  /* config options here */
  // ตั้งค่าให้เหมาะกับ Vercel
  trailingSlash: false,
  poweredByHeader: false,
  
  experimental: {
    // ลบ turbo config ที่ deprecated
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
  
  // ปิด ESLint สำหรับ POC
  eslint: {
    ignoreDuringBuilds: true,
  },
  
  // ปิด TypeScript type checking สำหรับ POC
  typescript: {
    ignoreBuildErrors: true,
  },
  
  // ปิด telemetry
  telemetry: false,
  
  // เพิ่ม rewrites สำหรับ Vercel
  async rewrites() {
    return [
      {
        source: '/(.*)',
        destination: '/',
      },
    ];
  },
};

export default nextConfig; 