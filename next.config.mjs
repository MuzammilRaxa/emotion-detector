/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',

  env: {
    NEXT_PUBLIC_WS_URL: process.env.NODE_ENV === 'production'
      ? 'wss://emotion-tracker-py-production.up.railway.app'  // ← CHANGE to wss:// and remove trailing slash
      : 'ws://localhost:8765'
      // : 'wss://emotion-tracker-py-production.up.railway.app'
  },

  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self' 'unsafe-eval' 'unsafe-inline' data: blob:; connect-src 'self' ws: wss:;"
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          }
        ],
      },
    ]
  }
}

export default nextConfig;